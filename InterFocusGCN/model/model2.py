import torch
import torch as th
import torch.nn.functional as F
from torch import nn

from .layer.gatedgcn_layer import HeteroGatedGCNLayer

class HeteroGatedGCN(nn.Module):
    def __init__(
            self,
            p_in_channels=41,
            p_edge_features=5,
            l_in_channels=41,
            l_edge_features=10,
            num_hidden_channels=128,
            residual=True,
            dropout_rate=0.1,
            equivstable_pe=False,
            num_layers=4,
            ):
        super(HeteroGatedGCN, self).__init__()
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.p_node_encoder = nn.Linear(p_in_channels, num_hidden_channels)
        self.p_edge_encoder = nn.Linear(p_edge_features, num_hidden_channels)
        self.l_node_encoder = nn.Linear(l_in_channels, num_hidden_channels)
        self.l_edge_encoder = nn.Linear(l_edge_features, num_hidden_channels)
        self.pl_edge_encoder = nn.Linear(p_in_channels+l_in_channels, num_hidden_channels  * 2 )
        gt_block_modules = [HeteroGatedGCNLayer(
                            num_hidden_channels,
                            num_hidden_channels,
                            dropout_rate,
                            residual,
                            equivstable_pe=equivstable_pe) for _ in range(num_layers)]
        self.gt_block = nn.ModuleList(gt_block_modules)

    def forward(self, g , p_node_feats, p_edge_feats  , l_node_feats , l_edge_feats , pl_edge_feats ):
        p_node_feats = self.p_node_encoder(p_node_feats)
        p_edge_feats = self.p_edge_encoder(p_edge_feats)
        l_node_feats = self.l_node_encoder(l_node_feats)
        l_edge_feats = self.l_edge_encoder(l_edge_feats)
        pl_edge_feats = self.pl_edge_encoder(pl_edge_feats)
        g['protein'].h = p_node_feats
        g['protein','p2p','protein'].h = p_edge_feats
        g['ligand'].h = l_node_feats
        g[ 'ligand' , 'l2l','ligand'].h = l_edge_feats
        g[ 'inter_edge' ].h = pl_edge_feats
        for gt_layer in self.gt_block:
            g = gt_layer(g)
        return g['protein'].h , g['ligand'].h , g['inter_edge' ].h

def get_batch_num_nodes(batch,ntype,device):
    return torch.bincount(batch[ntype].batch, minlength=batch.batch_size).to(device)

def to_dense_batch_dgl(bg, feats,  ntype , fill_value=0):
    bg_batch_info = bg[ntype].batch
    batch_size = bg_batch_info.max() + 1
    device = feats.device
    batch_num_nodes =  get_batch_num_nodes(bg,ntype,device)
    num_nodes = feats.shape[0]
    max_num_nodes = int(batch_num_nodes.max())
    batch = th.cat([th.full((1,x.type(th.int)), y) for x,y in zip(batch_num_nodes ,range(batch_size))],dim=1).reshape(-1).type(th.long).to(device)
    cum_nodes = th.cat([batch.new_zeros(1), batch_num_nodes.cumsum(dim=0)])
    idx = th.arange(num_nodes, dtype=th.long, device=device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    size = [batch_size * max_num_nodes] + list(feats.size())[1:]
    out = feats.new_full(size, fill_value)
    out[idx] = feats
    out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])
    mask = th.zeros(batch_size * max_num_nodes, dtype=th.bool,
                    device=device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)
    return out, mask

def batch_num_edges(batch , etype , batch_size  , device):
    src_type = etype[0]
    src_batch = batch[src_type].batch
    src_indices = batch[etype].edge_index[0]
    edge_batch = src_batch[src_indices]
    counts = torch.bincount(edge_batch, minlength=batch_size).to(device)
    return counts

def create_inter_edge( B , N_p , N_l ,  e_pl  , b_hetero_g, dtype):
    device = e_pl.device
    inter_edge = th.zeros(size=(B*N_p*N_l,e_pl.size(-1) ), dtype=dtype).to(device)
    feat = e_pl
    batch = th.cat([th.full((1, x.type(th.int)), y) for x, y in zip(  get_batch_num_nodes(b_hetero_g , ntype='inter_edge' , device=device) , range( B ))], dim=1).reshape(-1).type(th.long).to(device)
    p_idx = b_hetero_g['inter_edge'].src_dst[:,0]
    l_idx = b_hetero_g['inter_edge'].src_dst[:,1]
    main_idx=batch * N_p * N_l + p_idx * N_l + l_idx
    inter_edge[main_idx] = feat
    inter_edge = inter_edge.view( B , N_p , N_l , -1)
    return inter_edge

class InterFocusGCN(nn.Module):
    def __init__(self, hetero_model, in_channels, hidden_dim, n_gaussians, dropout_rate=0.15,
                    dist_threhold=1000):
        super(InterFocusGCN, self).__init__()
        self.hetero_model = hetero_model
        self.MLP = nn.Sequential(nn.Linear(in_channels * 4 , hidden_dim),
                                nn.BatchNorm1d(hidden_dim), 
                                nn.ELU(), 
                                nn.Dropout(p=dropout_rate)
                                ) 
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        self.atom_types = nn.Linear(in_channels, 17)
        self.bond_types = nn.Linear(in_channels*2, 4)
        self.dist_threhold = dist_threhold    
    
    def forward(self,  b_hetero_g):
        h_p, h_l , e_pl   = self.hetero_model(b_hetero_g ,
                                     b_hetero_g['protein'].feats.float(),
                                     b_hetero_g['protein','p2p','protein'].feats.float(),
                                     b_hetero_g['ligand'].feats.float(),
                                     b_hetero_g['ligand','l2l' ,'ligand'].feats.float() ,
                                     b_hetero_g['inter_edge'].feats.float()
                                     )
        h_p_x, p_mask = to_dense_batch_dgl(b_hetero_g, h_p,  'protein', fill_value=0)
        h_l_x, l_mask = to_dense_batch_dgl(b_hetero_g, h_l,  'ligand' ,  fill_value=0)
        h_p_pos, _ = to_dense_batch_dgl(b_hetero_g,b_hetero_g['protein'].pos,  'protein',  fill_value=0)
        h_l_pos, _ = to_dense_batch_dgl(b_hetero_g,b_hetero_g['ligand'].pos,   'ligand' , fill_value=0)
        (B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)
        self.B = B
        self.N_l = N_l
        self.N_p = N_p
        h_l_x = h_l_x.unsqueeze(-3)
        h_l_x = h_l_x.repeat(1, N_p ,1  , 1)
        h_p_x = h_p_x.unsqueeze(-2)
        h_p_x = h_p_x.repeat(1, 1, N_l, 1)
        C = th.cat((h_l_x, h_p_x), -1)
        inter_edge = create_inter_edge( B , N_p , N_l  ,  e_pl  , b_hetero_g , C.dtype)
        C = th.concatenate([C , inter_edge] , dim = -1 )
        self.C_mask = C_mask = l_mask.view(B, 1, N_l) & p_mask.view(B, N_p, 1)
        self.C = C = C[C_mask]
        C = self.MLP(C)
        C_batch = th.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_p, N_l)
        if C_mask.is_cuda:
            C_batch = C_batch.to('cuda')
        C_batch = C_batch[C_mask]
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1
        atom_types = self.atom_types(h_l)
        bond_types = self.bond_types(th.cat([h_l[ b_hetero_g['ligand' , 'l2l' ,'ligand'].edge_index[0]],h_l[b_hetero_g ['ligand','l2l','ligand'].edge_index[1]]], axis=1))
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_p_pos.view(B,-1,3) , h_p_pos.size(2)  )
        dist =  th.permute(dist , (0 , 2, 1 ))
        dist = dist[C_mask]
        return pi, sigma, mu, dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch
    
    def compute_euclidean_distances_matrix(self, X, Y , atom_num_per_res):
        X = X.double()
        Y = Y.double()
        dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y**2,    axis=-1).unsqueeze(1) + th.sum(X**2, axis=-1).unsqueeze(-1)
        return th.nan_to_num((dists**0.5).view(self.B, self.N_l,-1,atom_num_per_res),10000).min(axis=-1)[0]

