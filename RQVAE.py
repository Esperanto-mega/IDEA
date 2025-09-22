import torch 
import torch.nn as nn
from torch.nn import functional as F
from geomloss import SamplesLoss
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters = num_clusters, max_iter = num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(logits, temperature = 1., stochastic = False, 
    straight_through = False, dim = -1, training = True):
    
    dtype, size = logits.dtype, logits.shape[dim]

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)
    one_hot = F.one_hot(ind, size).type(dtype)

    if not straight_through or temperature <= 0. or not training:
        return ind, one_hot

    π1 = (logits / temperature).softmax(dim = dim)
    one_hot = one_hot + π1 - π1.detach()

    return ind, one_hot

class VQ_embedding(nn.Module):
    def __init__(self, codebook_size, codebook_dims,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 sk_epsilon = 0.003, sk_iters = 100,):
        super(VQ_embedding, self).__init__()
        self.codebook_size = codebook_size
        self.codebook_dims = codebook_dims
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dims)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape = None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-3
        if amplitude <= 0:
            print(amplitude)
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk = True):
        # Flatten input
        latent = x.view(-1, self.codebook_dims)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
            2 * torch.matmul(latent, self.embedding.weight.t())
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            # d = d.float()  # Ensure d is float for Sinkhorn algorithm
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d

class RQ_embedding(nn.Module):
    def __init__(self, n_e_list, codebook_dims, sk_epsilons = None, beta = 0.25,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100):
        super(RQ_embedding,self).__init__()
        self.n_e_list = n_e_list
        self.codebook_dims = codebook_dims
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons if sk_epsilons is not None else [0.003] * self.num_quantizers
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VQ_embedding(codebook_size, codebook_dims,
                                                        beta=self.beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for codebook_size, sk_epsilon in zip(n_e_list,self.sk_epsilons) ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res # residual link
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices

class RV_Embedding(nn.Module):
    def __init__(self, rq_quantizers, codebook_size , codebook_dims):
        super(RV_Embedding, self).__init__()
        self.quantizer_num = rq_quantizers # H
        self.codebook_size = codebook_size # K
        self.codebook_dims = codebook_dims # D 
        if isinstance(rq_quantizers,list):
            rq_layers = rq_quantizers
        else :
            rq_layers = [codebook_size] * rq_quantizers
        self.codebook_rq = RQ_embedding(rq_layers, codebook_dims, None)
        self.codebook_vq = VQ_embedding(codebook_size, codebook_dims)
    def forward(self, x, y):
        # x: explanation embedding
        # y: original embedding
        
        # x =  y = shape[1,D]
        # 给exp embedding做2次VQ
        x_q, rq_loss, _ = self.codebook_rq(x)
        
        # x_q = shape[1,D]
        # 给quantized exp embedding和ori embedding做1次VQ
        x_hat, vq_loss_1, _ = self.codebook_vq(x_q) 
        y_hat, vq_loss_2, _ = self.codebook_vq(y)

        # 等价于exp embedding做3次VQ，ori embedding做1次VQ
        
        return x_hat, y_hat, rq_loss + vq_loss_1 + vq_loss_2 
    
    def get_vq_codebook(self):
        return self.codebook_vq.get_codebook()
    
    def get_rq_codebook(self):
        return self.codebook_rq.get_codebook()

class MLP_Decoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        super(MLP_Decoder, self).__init__()
        self.layers = nn.Sequential()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.layers.add_module(f"linear_{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # add normalize
                #self.layers.add_module(f"batchnorm_{i}", nn.BatchNorm1d(layer_sizes[i + 1]))
                self.layers.add_module(f"dropout_{i}", nn.Dropout(p=dropout_prob))
                self.layers.add_module(f"relu_{i}", nn.ReLU())

    def forward(self, x):
        #print(x.shape)
        return self.layers(x)
    
class RV_model(nn.Module):
    def __init__(self, codebook_size = 64, codebook_dims = 32):
        super(RV_model,self).__init__()
        
        self.base_codebook = VQ_embedding(codebook_size, codebook_dims)
        self.exp_codebook = VQ_embedding(codebook_size, codebook_dims)
        
        self.decoder = MLP_Decoder(input_size = codebook_dims,
                                  hidden_sizes = [128],
                                  output_size = 2)
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        
        self.temperature = 4
        
        self.alpha = 0.65
        
    def forward(self, es, eo):
        eo_quantized, loss_eo, _, _ = self.base_codebook(eo)
        eo_residual = eo - eo_quantized
        eo_residual_quantized, loss_eo_residual, _, dis_eo_residual = self.exp_codebook(eo_residual)
        es_quantized, loss_es, _, dis_es = self.exp_codebook(es)
        
        dis_eo_residual = torch.softmax(dis_eo_residual, dim = -1)
        dis_es = torch.softmax(dis_es, dim = -1)
        W_loss = SamplesLoss(loss = 'sinkhorn', p = 2, blur = 0.05, scaling = 0.5)
        distribution_loss = W_loss(dis_eo_residual, dis_es)
        distribution_loss += 0.5 *(W_loss(dis_eo_residual, dis_eo_residual) + W_loss(dis_es, dis_es))

        eo_rec_loss = self.mse_loss(eo_quantized + eo_residual_quantized, eo)
        es_rec_loss = self.mse_loss(es_quantized, es)
        reconstruction_loss = eo_rec_loss + es_rec_loss
        
        total_loss = loss_eo + loss_es + loss_eo_residual + distribution_loss + reconstruction_loss
        
        return total_loss, distribution_loss, reconstruction_loss

    def get_exp_codebook(self):
        return self.rv_emd.codebook_rq.get_codebook()
    
    def get_base_codebook(self):
        return self.rv_emd.codebook_vq.get_codebook()
