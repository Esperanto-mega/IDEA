import torch
from RQVAE import VQ_embedding
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
class GraphTokenizer(torch.nn.Module):
    def __init__(self, codebook_size, codebook_dims, decoder_dims,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 50,
                 sk_epsilon = 0.003, sk_iters = 100, quant_loss_weight = 1.0,
                 rec_loss_type = 'mse', node_weight = 0.001, edge_weight = 0.03,
                 rec_loss_weight = 1.0):
        super(GraphTokenizer, self).__init__()
        self.codebook_size = codebook_size
        self.codebook_dims = codebook_dims
        self.decoder_dims = decoder_dims
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.quant_loss_weight = quant_loss_weight
        self.rec_loss_type = rec_loss_type
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.rec_loss_weight = rec_loss_weight
        
        self.codebook_a = VQ_embedding(
            codebook_size = codebook_size,
            codebook_dims = codebook_dims,
            beta = beta,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sk_epsilon = sk_epsilon,
            sk_iters = sk_iters
        )
        self.codebook_b = VQ_embedding(
            codebook_size = codebook_size,
            codebook_dims = codebook_dims,
            beta = beta,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sk_epsilon = sk_epsilon,
            sk_iters = sk_iters
        )
        
        self.decoder_node = torch.nn.Linear(codebook_dims, decoder_dims)
        self.decoder_edge = torch.nn.Linear(codebook_dims, codebook_dims)
        
    def forward(self, embedding, x, edge_index, use_sk = False):
        adj = to_dense_adj(edge_index.long(), max_num_nodes = x.size(0)).squeeze(0)
        x_q = 0
        residual = embedding
        
        x_res_a, loss_a, _, _ = self.codebook_a(residual, use_sk = use_sk)
        x_q = x_q + x_res_a
        residual = residual - x_res_a
        
        x_res_b, loss_b, _, _ = self.codebook_b(residual, use_sk = use_sk)
        x_q = x_q + x_res_b
        residual = residual - x_res_b
        
        quantized_node = self.decoder_node(x_q)
        quantized_edge = self.decoder_edge(x_q)
        
        quantized_adj = torch.matmul(quantized_edge, quantized_edge.t())
        quantized_adj = (quantized_adj - quantized_adj.min()) / (quantized_adj.max() - quantized_adj.min() + 1e-8)
        
        rq_loss = (loss_a + loss_b) / 2
        
        if self.rec_loss_type == 'mse':
            emb_rec_loss = F.mse_loss(x_q, embedding, reduction = 'mean') * self.rec_loss_weight
            node_rec_loss = F.mse_loss(quantized_node, x, reduction = 'mean') * self.node_weight
        elif self.rec_loss_type == 'l1':
            emb_rec_loss = F.l1_loss(x_q, embedding, reduction = 'mean') * self.rec_loss_weight
            node_rec_loss = F.l1_loss(quantized_node, x, reduction = 'mean') * self.node_weight
        
        edge_rec_loss = torch.sqrt(F.mse_loss(quantized_adj, adj, reduction = 'mean')) * self.edge_weight
        
        rq_loss = rq_loss * self.quant_loss_weight
        
        rqvae_loss = emb_rec_loss + node_rec_loss + edge_rec_loss + rq_loss
        
        return x_res_a, x_res_b, rqvae_loss, emb_rec_loss, node_rec_loss, edge_rec_loss, rq_loss