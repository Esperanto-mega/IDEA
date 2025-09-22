import logging
from typing import Optional, Union
import sys
sys.path.append('..')
import torch
from torch import Tensor
from torch.nn import ReLU, Sequential

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import (
    ExplanationType,
    ModelMode,
    ModelTaskLevel,
)
from torch_geometric.nn import Linear,global_mean_pool
from torch_geometric.nn.inits import reset
from torch_geometric.utils import get_embeddings
from RQVAE import RV_model

class Idea(ExplainerAlgorithm):

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0.01,
        'emd': 1.0
    }

    def __init__(self, epochs: int, lr: float = 0.003,codebook_size=32,codebook_channels=32, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = Sequential(
            Linear(-1, 32),
            ReLU(),
            Linear(32, 1),
        )
        self.rv_emd_model = RV_model(codebook_size = codebook_size, codebook_dims = codebook_channels)
        self._curr_epoch = -1
        self.optimizer = torch.optim.Adam([
            {'params': self.mlp.parameters(),'lr': lr},  
            {'params': self.rv_emd_model.parameters(), 'lr': 0.1}
        ])
        self.coeffs.update(kwargs)
        
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    def train(
        self,
        epoch: int,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        emd_target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ):
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

        z = get_embeddings(model, x, edge_index, **kwargs)[-1]
        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)

        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)
        edge_mask = self._concrete_sample(logits, temperature)
        set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

        if self.model_config.task_level == ModelTaskLevel.node:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))
            edge_mask = edge_mask[hard_edge_mask]

        emd = model.embedding(x, edge_index)
        if index is not None:
            emd = emd[index]
        else:
            emd = global_mean_pool(emd, **kwargs)
        rqvae_loss, _distr_loss, _recon_loss = self.rv_emd_model(emd, emd_target)
        rqvae_loss *= self.coeffs['emd'] # Adjust the weight of the RQVAE loss as needed

            
        rqvae_loss.backward()
        self.optimizer.step()

        clear_masks(model)
        self._curr_epoch = epoch

        return (
            float(rqvae_loss),
            float(_distr_loss), 
            float(_recon_loss)
        )

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        if self._curr_epoch < self.epochs - 1:  # Safety check:
            raise ValueError(f"'{self.__class__.__name__}' is not yet fully "
                             f"trained (got {self._curr_epoch + 1} epochs "
                             f"from {self.epochs} epochs). Please first train "
                             f"the underlying explainer model by running "
                             f"`explainer.algorithm.train(...)`.")

        hard_edge_mask = None
        if self.model_config.task_level == ModelTaskLevel.node:
            if index is None:
                raise ValueError(f"The 'index' argument needs to be provided "
                                 f"in '{self.__class__.__name__}' for "
                                 f"node-level explanations")
            if isinstance(index, Tensor) and index.numel() > 1:
                raise ValueError(f"Only scalars are supported for the 'index' "
                                 f"argument in '{self.__class__.__name__}'")

            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))


        clear_masks(model)
        z = get_embeddings(model, x, edge_index, **kwargs)[-1]
        inputs = self._get_inputs(z, edge_index, index)
        logits = self.mlp(inputs).view(-1)

        edge_mask = self._post_process_mask(logits, hard_edge_mask,
                                            apply_sigmoid=True)
        
        # subgraph embedding
        set_masks(model, edge_mask, edge_index, apply_sigmoid=False)
        emd = model.embedding(x, edge_index)
        if index is not None:
            emd = emd[index]
        else:
            emd = global_mean_pool(emd, batch = kwargs.get('batch', None))
        clear_masks(model)
        
        return Explanation(edge_mask=edge_mask, subgraph_embedding=emd)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"phenomenon explanations "
                          f"got (`explanation_type={explanation_type.value}`)")
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"node-level or graph-level explanations "
                          f"got (`task_level={task_level.value}`)")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(f"'{self.__class__.__name__}' does not support "
                          f"explaining input node features "
                          f"got (`node_mask_type={node_mask_type.value}`)")
            return False

        return True

    ###########################################################################

    def _get_inputs(self, embedding: Tensor, edge_index: Tensor,
                    index: Optional[int] = None) -> Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        if self.model_config.task_level == ModelTaskLevel.node:
            assert index is not None
            zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: Tensor,
                         temperature: float = 1.0) -> Tensor:
        
        
        bias = self.coeffs['bias']
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _loss(self, y_hat: Tensor, y: Tensor, edge_mask: Tensor) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs['edge_ent']

        loss_sum = loss + size_loss + mask_ent_loss
        return loss_sum,loss ,size_loss ,mask_ent_loss