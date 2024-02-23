
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn

class LOFGCN(torch.nn.Module):
    
    def __init__(self, num_genes, edge_index, hidden_size, 
                 dropout=0.106, activation_func=torch.nn.SELU(), num_classes=200):
                 super(LOFGCN, self).__init__()
                 self.num_genes = num_genes
                 self.edge_index = edge_index
                 self.gene_emb = nn.Embedding(num_embeddings=num_genes, embedding_dim=hidden_size-1, )
                 self.bn_emb = nn.BatchNorm1d(num_features=num_genes)
                 self.layers = torch.nn.ModuleList()
                 self.activation_func = activation_func
                 self.dropout = torch.nn.Dropout(dropout)
                 self.conv1 = GCNConv(hidden_size, 64)
                 self.conv2 = GCNConv(64, 32)
                 self.lin = torch.nn.Linear(32, num_classes)
                 
    def forward(self, batch):

            num_genes = torch.tensor(self.num_genes).to('cuda')
            edge_index = torch.tensor(self.edge_index).to('cuda')
            deg_tensor = torch.tensor(batch, dtype=torch.float32).to('cuda')

            # Embedding layer
            longtensor = torch.LongTensor(list(range(num_genes))).repeat(batch.shape[0], 1).to('cuda')
            emb = self.gene_emb(longtensor)
            emb = torch.cat((emb, deg_tensor), dim=-1).to('cuda')
            x = self.bn_emb(emb)
            
            # 1st GCN layer
            x = self.conv1(x, edge_index)
            x = self.activation_func(x)
            x = self.dropout(x)

            # 2nd GCN layer
            x = self.conv2(x, edge_index)
            x = self.activation_func(x)
            x = self.dropout(x)

            # Readout Layer
            x = torch.mean(x, dim=1)
            x = self.lin(x)      

            return x