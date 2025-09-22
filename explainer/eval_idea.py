from tqdm import tqdm
import argparse
import random
import torch
import numpy as np
from time import time
import logging
import os
import sys
import torch.nn.functional as F
import pickle

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer
from torch_geometric.nn import global_mean_pool

from gnn_model import *
from idea import Idea
from sklearn.metrics import roc_auc_score

def topk_mask_py(lst, k):
    if k <= 0:
        return [0] * len(lst)
    indexed = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)
    mask = [0] * len(lst)
    for idx, _ in indexed[:k]:
        mask[idx] = 1
    return mask

def get_dataset(dataset_name="BA2Motif", root_path=".", **kwargs):
    if dataset_name == "Mutag":
        with open(root_path + 'data/mutag_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_name == "Benzene":
        with open(root_path + 'data/benzene_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_name == "Alkane":
        with open(root_path + 'data/alkane_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_name == "Fluoride":
        with open(root_path + 'data/fluoride_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    elif dataset_name == "BA2Motif":
        with open(root_path + 'data/ba2motifs_with_gt.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return dataset

def init_model(dataset_name="BA2Motif", root_path = ".", device = None):
    model_path = root_path + f'GNN/{dataset_name}.pth'
    model = torch.load(model_path, weights_only=False, map_location=torch.device(device))
    model.eval()
    return model

def train_explainer(explainer, train_loader, epochs = 20,
                    exp_type = "phenomenon", device = None):
    losses = []
    start_time = time()
    
    emd_targets = []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            explainer.model.eval()
            emd = explainer.model.embedding(data.x, data.edge_index)
            emd_targets.append(global_mean_pool(emd, data.batch))
    
    for epoch in range(epochs):
        epoch_loss = {
            'rqvae_loss': 0.0,
            'distribution_loss': 0.0,
            'codebook_loss': 0.0,
        }
        loss_key = list(epoch_loss.keys())
        for idx, data in enumerate(train_loader):
            data = data.to(device)
            target = explainer.model(data.x, data.edge_index, data.batch)
            ls = explainer.algorithm.train(epoch, 
                            explainer.model, 
                            data.x, 
                            data.edge_index, 
                            target = target,
                            emd_target = emd_targets[idx], 
                            batch = data.batch)
            assert len(ls) == len(loss_key), "Loss keys and values mismatch"
            for i in range(len(loss_key)):
                epoch_loss[loss_key[i]] += ls[i]
        avg_loss = {k: v / len(train_loader) for k, v in epoch_loss.items()}
        logging.info(f"Epoch {epoch+1}/{epochs}"
                        f"- RQVAE Loss: {avg_loss['rqvae_loss']:.4f}\n" +
                        f"- Codebook Loss: {avg_loss['codebook_loss']:.4f}\n" +
                        f"- Distribution Loss: {avg_loss['distribution_loss']:.4f}\n" +
                        f"- Time: {time() - start_time:.2f}s")
        losses.append(avg_loss['rqvae_loss'])

    return losses

def evaluate_explainer(explainer, test_loader, model, device=None, gt_mask=None, dataset = None):
    results = {
        'roc_auc': [],
    }
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc = "Processing Test Data", ncols = 80)):
            data = data.to(device)
            graph_embedding = explainer.model.embedding(data.x, data.edge_index)
            graph_embedding = global_mean_pool(graph_embedding, data.batch)
            target = explainer.model(data.x, data.edge_index, data.batch)
            exp = explainer(data.x, data.edge_index, target = target, batch = data.batch)
            
            same = (data.edge_mask == data.edge_mask[0]).all().item()
            if not same:
                auc = roc_auc_score(data.edge_mask.int().tolist(), exp.edge_mask.cpu().tolist())
                results['roc_auc'].append(auc)
    
    logging.info(f"Evaluation Results \n"
                 f"- ROC AUC: {np.mean(results['roc_auc']):.4f}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="PGExplainer Test")
    parser.add_argument("--root_path", type=str, default="", help="Path to IDEA directory")
    parser.add_argument("--dataset", type=str, default="BA2Motif", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type = int, default = 0, help = "GPU ID to use (-1 for CPU)")
    parser.add_argument("--seed", type= int, default=2025, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--codebook_size", type=int, default=32, help="Size of the codebook for enhanced PGExplainer")
    
    parser.add_argument('--use_ckpt', type=int, default=1)
    parser.add_argument("--gt_learning_rate", type=float, default=1e-3, help="GT Learning rate")
    parser.add_argument("--gt_epochs", type=int, default=10, help="Training epochs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # if args.device >= 0 and torch.cuda.is_available():
    #     device = f'cuda:{args.device}'
    if args.device >= 0 and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logging.info(f"Using device: {device}")

    if args.dataset == "Mutag":
        dataset = get_dataset(args.dataset, args.root_path)
        train_set, test_set = dataset[800:], dataset[:800]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(train_set) + len(test_set)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    elif args.dataset == "Benzene":
        dataset = get_dataset(args.dataset, args.root_path)
        test_num = int(0.3 * len(dataset))
        train_set, test_set = dataset[test_num:], dataset[:test_num]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    elif args.dataset == "Alkane":
        dataset = get_dataset(args.dataset, args.root_path)
        train_set, test_set = dataset[500:], dataset[:500]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    elif args.dataset == "Fluoride":
        dataset = get_dataset(args.dataset, args.root_path)
        train_set, test_set = dataset[500:], dataset[:500]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    elif args.dataset == 'BA2Motif':
        dataset = get_dataset(args.dataset, args.root_path)
        train_set, test_set = dataset[:800], dataset[800:]
        num_features = train_set[0].x.shape[1]
        num_classes = (torch.unique(torch.tensor([data.y for data in test_set]))).numel()
        logging.info(f"Loaded {args.dataset} dataset with {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        logging.info(f"Training graphs: {len(train_set)}, Testing graphs: {len(test_set)}")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    explained_model = init_model(args.dataset, args.root_path, device)
    logging.info(f"Loaded pretrained GNN from model {args.root_path + f'models/{args.dataset}.pt'}")

    model_config = {
        'mode': "multiclass_classification" if num_classes > 2 else "binary_classification",
        'task_level': "graph",
        'return_type': "raw"
    }
    __embedding = explained_model.embedding(train_set[0].x.to(device), train_set[0].edge_index.to(device))
    explainer = Explainer(
        model = explained_model,
        algorithm = Idea(
            epochs = args.epochs, lr = args.lr, 
            codebook_size = args.codebook_size, codebook_channels = __embedding.shape[1]
        ).to(device),
        explanation_type = "phenomenon",
        edge_mask_type = "object",
        model_config = model_config
    )

    codebook_root = args.root_path + 'codebook/'
    if args.use_ckpt and os.path.isfile(codebook_root + f'Codebook_A_{args.dataset}.pt'):
        logging.info(f'Loading pretrained codebooks from {codebook_root}')
        pretrained_weight = torch.load(codebook_root + f'Codebook_A_{args.dataset}.pt')
        explainer.algorithm.rv_emd_model.base_codebook.embedding.weight.data = pretrained_weight.to(device)
        pretrained_weight = torch.load(codebook_root + f'Codebook_B_{args.dataset}.pt')
        explainer.algorithm.rv_emd_model.exp_codebook.embedding.weight.data = pretrained_weight.to(device)
    else:
        from graph_tokenizer import GraphTokenizer
        graph_tokenizer = GraphTokenizer(
            codebook_size = args.codebook_size,
            codebook_dims = __embedding.shape[1],
            decoder_dims = num_features,
        ).to(device)
        
        __optimizer = torch.optim.Adam(
            graph_tokenizer.parameters(),
            lr = args.gt_learning_rate if hasattr(args, 'gt_learning_rate') else 1e-3,
            weight_decay = 1e-5
        )
        __criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(args.gt_epochs):
            graph_tokenizer.train()
            total_loss = 0
            token_losses, emb_rec_losses, node_rec_losses, edge_rec_losses, rq_losses = 0, 0, 0, 0, 0
            a_losses, b_losses = 0, 0
            correct = 0
            bst_correct = 0
            for data in tqdm(train_loader):
                data = data.to(device)
                __optimizer.zero_grad()
                
                embed = explained_model.embedding(data.x, data.edge_index)
                
                x_res_a, x_res_b, token_loss_all, emb_rec_loss, node_rec_loss, edge_rec_loss, rq_loss = graph_tokenizer(
                    embed, data.x, data.edge_index, use_sk = True)
                
                prediction_a = explained_model.prediction(x_res_a, batch = data.batch)
                prediction_b = explained_model.prediction(x_res_b, batch = data.batch)
                
                uniform_target = torch.ones_like(prediction_a, dtype=torch.float) / prediction_a.size(1)
                
                loss_a = F.kl_div(F.log_softmax(prediction_a, dim=1), uniform_target, reduction='batchmean')
                loss_b = __criterion(prediction_b, data.y)
                
                final_loss = token_loss_all + loss_a + loss_b
                
                final_loss.backward()
                __optimizer.step()
                
                total_loss += final_loss.item()
                token_losses += token_loss_all.item()
                emb_rec_losses += emb_rec_loss.item()
                node_rec_losses += node_rec_loss.item()
                edge_rec_losses += edge_rec_loss.item()
                rq_losses += rq_loss.item()
                a_losses += loss_a.item()
                b_losses += loss_b.item()
                correct += (prediction_b.argmax(dim = 1) == data.y).sum().item()
                
                if correct > bst_correct:
                    bst_correct = correct
                    codebook_a_weight = graph_tokenizer.codebook_a.embedding.weight.data.cpu()
                    codebook_b_weight = graph_tokenizer.codebook_b.embedding.weight.data.cpu()
            
            logging.info(f'Epoch {epoch + 1}/{args.gt_epochs}, Loss: {total_loss / len(train_loader)}')
            logging.info(f'Token Loss: {token_losses / len(train_loader)}, Embed Rec Loss: {emb_rec_losses / len(train_loader)}, Node Rec Loss: {node_rec_losses / len(train_loader)}, Edge Rec Loss: {edge_rec_losses / len(train_loader)}, RQ Loss: {rq_losses / len(train_loader)}')
            logging.info(f'A Loss: {a_losses / len(train_loader)}, B Loss: {b_losses / len(train_loader)}')
            logging.info(f'Accuracy: {correct / len(train_set)}')
            
            torch.save(codebook_a_weight, 
                        codebook_root + f'Codebook_A_{args.dataset}.pt')
            torch.save(codebook_b_weight, 
                        codebook_root + f'Codebook_B_{args.dataset}.pt')
        
        explainer.algorithm.rv_emd_model.base_codebook.embedding.weight.data = codebook_a_weight.to(device)
        explainer.algorithm.rv_emd_model.exp_codebook.embedding.weight.data = codebook_b_weight.to(device)

    _ = train_explainer(explainer,
                        train_loader, 
                        args.epochs,
                        device = device,
    )
    _ = evaluate_explainer(explainer,
                                    test_loader,
                                    explained_model,
                                    device = device,
                                    gt_mask = True,
                                    dataset = args.dataset)

if __name__ == '__main__':
    main()