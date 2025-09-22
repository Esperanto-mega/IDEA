import torch
from torch_geometric.nn import (
    GCNConv, 
    GINConv, 
    BatchNorm, 
    SAGEConv, 
    JumpingKnowledge, 
    GATConv, 
    Sequential, 
    global_mean_pool,
    global_max_pool
)
from torch.nn import ReLU, Linear
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score, roc_auc_score
import numpy as np 

class NodeGCN_3layer(torch.nn.Module):
    def __init__(self, in_channels, classes,hidden_channels):
        super(NodeGCN_3layer, self).__init__()
        self.embedding_size = hidden_channels * 3
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size, classes)

    def forward(self, x, edge_index, edge_weights=None):
        self.device = x.get_device()
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin
    
class NodeGCN_2layer(torch.nn.Module):
    def __init__(self, in_channels, classes,hidden_channels):
        super(NodeGCN_2layer, self).__init__()
        self.embedding_size = hidden_channels * 2
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.relu2 = ReLU()
        self.lin = Linear(self.embedding_size, classes)

    def forward(self, x, edge_index, edge_weights=None):
        self.device = x.get_device()
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class GraphGCN(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, num_classes):
        super(GraphGCN, self).__init__()
        self.embedding_size = 20
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size * 2, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long) ##artificat without device specification
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)) ##artificat without device specification
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin

    def graph_embedding(self, x, edge_index, batch=None, edge_weights=None): #this is the same as forward()
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long)
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        return input_lin

class GraphGIN(torch.nn.Module):
    """
    A GIN model using 3 layers of GIN
    """
    def __init__(self, num_feats, num_classes):
        super().__init__()
        hidden_channels = 20
        self.mlp_gin1 = torch.nn.Linear(num_feats, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin3 = GINConv(self.mlp_gin3)
        self.lin = Linear(hidden_channels*2, num_classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long) ##artificat without device specification
        embed = self.embedding(x, edge_index)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index):
        x = self.gin1(x = x, edge_index = edge_index)
        x = x.relu()
        x = self.gin2(x = x, edge_index = edge_index)
        x = x.relu()
        x = self.gin3(x = x, edge_index = edge_index)
        x = x.relu()
        return x

class GCN(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.relu3 = ReLU()
        self.lin = Linear(hidden_channels * 2, classes)

    def forward(self, x, edge_index, batch=None, edge_weights=None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim=-1)

        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1)).to(x.device)
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin

class GCN_1layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, classes):
        super(GCN_1layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, classes)

    def forward(self, x, edge_index, batch):
        x = self.embedding(x,edge_index,batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x

    def embedding(self,x,edge_index,batch):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = global_mean_pool(x, batch)
        return x

class GCN_2layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, classes):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, classes)

    def forward(self, x, edge_index, batch=None):
        x = self.embedding(x,edge_index)
        if batch != None:
            x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x
    
    def prediction(self, x, batch = None):
        if batch is not None:
            x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x) 
        return x
    
    def forward_explain(self, x, edge_index, edge_mask = None, batch = None):
        x = self.embedding_explain(x, edge_index, edge_mask)
        if batch != None:
            x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x
    
    def embedding_explain(self, x, edge_index, edge_mask = None, EPS = 1):
        edge_mask = (edge_mask * EPS).sigmoid()
        x = self.conv1(x, edge_index, edge_weight = edge_mask)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight = edge_mask)
        x = self.batchnorm2(x)
        return x

    def embedding(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.batchnorm2(x)
        return x

class GCN_3layer(torch.nn.Module):
    def __init__(self, hidden_channels, in_channels, classes):
        super(GCN_3layer, self).__init__()
        
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.gcn3 = GCNConv(hidden_channels, hidden_channels)
        self.batchnorm3 = BatchNorm(hidden_channels)
        self.lin = Linear(hidden_channels, classes)

    def forward(self, x, edge_index, batch = None):
        x = self.embedding(x,edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        # x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x) 
        return x

    def embedding(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gcn2(x, edge_index)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gcn3(x, edge_index)
        x = self.batchnorm3(x)
        return x
    
    def prediction(self, x, batch = None):
        if batch is not None:
            x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x) 
        return x
    
    def forward_explain(self, x, edge_index, edge_mask = None, batch = None):
        x = self.embedding_explain(x, edge_index, edge_mask)
        if batch != None:
            x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x
    
    def embedding_explain(self, x, edge_index, edge_mask = None, EPS = 1):
        edge_mask = (edge_mask * EPS).sigmoid()
        x = self.gcn1(x, edge_index, edge_weight = edge_mask)
        x = self.batchnorm1(x)
        x = x.relu()
        x = self.gcn2(x, edge_index, edge_weight = edge_mask)
        x = self.batchnorm2(x)
        x = x.relu()
        x = self.gcn3(x, edge_index, edge_weight = edge_mask)
        x = self.batchnorm3(x)
        return x

def split_dataset(dataset , split_size = 0.8, node_split = False):
    if not node_split:
        train_size = int( len(dataset) * split_size)
        dataset = dataset.shuffle()
        train_set = dataset[:train_size]
        test_set = dataset[train_size:]
    else :
        train_size = int( dataset[0].x.shape[0] * split_size)
        indics = torch.randperm(dataset[0].x.shape[0])
        train_set = indics[:train_size]
        test_set = indics[train_size:]
    return train_set,test_set

def train(loader ,model , optimizer,criterion, used_edge_attr=False):
    model.train()
    losses = 0.0
    correct = 0
    for data in loader :
        #print(data.x.shape,data.edge_index.shape)
        if used_edge_attr:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        else :
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        losses += loss.item()
        correct += int((out.argmax(dim=1) == data.y).sum())

    return model,losses,correct

def test(data_loader ,model,classes = None,used_edge_attr=False):
    with torch.no_grad():
        model.eval()
        if classes is None:
            classes = data_loader.dataset.num_classes # num_class => 2

        GT = [] # Groud truth 
        preds = [] # prediction
        probas = [] # probability
        correct = 0 # acc

        index = 0
        for data in data_loader:
            if used_edge_attr:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            else :
                out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            pred = out.argmax(dim=1)
            GT.extend(data.y) 
            preds.extend(pred)
            probas.extend(out.softmax(dim=1).squeeze()[:,1].detach().clone().cpu().numpy())
            correct += int((pred == data.y).sum())
            index += 1
        return correct