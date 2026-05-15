import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter
from torch_scatter import scatter_add

class HeteroGatedGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.p_A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.p_B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.p_C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.p_D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.p_E = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.l_A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.l_B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.l_C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.l_D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.l_E = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.inter_p_B = pyg_nn.Linear(in_dim, out_dim * 2, bias=True)
        self.inter_p_E = pyg_nn.Linear(in_dim, out_dim * 2, bias=True)
        self.inter_l_B = pyg_nn.Linear(in_dim, out_dim * 2, bias=True)
        self.inter_l_E = pyg_nn.Linear(in_dim, out_dim * 2, bias=True)
        self.inter_A = pyg_nn.Linear(in_dim*2, out_dim*2, bias=True)
        self.inter_D = pyg_nn.Linear(in_dim*2, out_dim*2, bias=True)
        self.p_bn_node_x = nn.BatchNorm1d(out_dim)
        self.p_bn_edge_e = nn.BatchNorm1d(out_dim)
        self.l_bn_node_x = nn.BatchNorm1d(out_dim)
        self.l_bn_edge_e = nn.BatchNorm1d(out_dim)
        self.inter_bn_edge_e = nn.BatchNorm1d(out_dim*2)
        self.dropout = dropout
        self.residual = residual

    def forward(self, batch):
        p_x, p_e, p_edge_index = batch['protein'].h, batch['protein','p2p','protein'].h, batch['protein' , 'p2p'  , 'protein'].edge_index
        l_x, l_e, l_edge_index = batch['ligand'].h, batch['ligand','l2l','ligand'].h, batch['ligand','l2l','ligand'].edge_index
        pl_x = batch['inter_edge'].h
        p2e_edge_index = batch['protein' ,'p2e' , 'inter_edge'].edge_index
        l2e_edge_index = batch['ligand' , 'l2e' , 'inter_edge'].edge_index
        if self.residual:
            p_x_in = p_x
            p_e_in = p_e
            l_x_in = l_x
            l_e_in = l_e
            pl_x_in = pl_x
        p_Ax = self.p_A(p_x)
        p_Bx = self.p_B(p_x)
        p_Ce = self.p_C(p_e)
        p_Dx = self.p_D(p_x)
        p_Ex = self.p_E(p_x)
        l_Ax = self.l_A(l_x)
        l_Bx = self.l_B(l_x)
        l_Ce = self.l_C(l_e)
        l_Dx = self.l_D(l_x)
        l_Ex = self.l_E(l_x)
        pl_Ax =self.inter_A(pl_x)
        pl_Dx =self.inter_D(pl_x)
        pl_p_Bx = self.inter_p_B(p_x)
        pl_p_Ex = self.inter_p_E(p_x)
        pl_l_Bx = self.inter_l_B(l_x)
        pl_l_Ex = self.inter_l_E(l_x)
        p_x, p_e = self.inner_edge_process(p_edge_index,
                              Bx=p_Bx, Dx=p_Dx, Ex=p_Ex, Ce=p_Ce,Ax=p_Ax)
        l_x , l_e = self.inner_edge_process(l_edge_index,
                              Bx=l_Bx, Dx=l_Dx, Ex=l_Ex, Ce=l_Ce, Ax=l_Ax)
        pl_x = self.inter_edge_process(p2e_edge_index,l2e_edge_index,
                             p_Bx = pl_p_Bx,l_Bx=pl_l_Bx, pl_Dx=pl_Dx, p_Ex = pl_p_Ex , l_Ex =  pl_l_Ex,pl_Ax=pl_Ax)
        p_x = self.p_bn_node_x(p_x)
        p_e = self.p_bn_edge_e(p_e)
        l_x = self.l_bn_node_x(l_x)
        l_e = self.l_bn_edge_e(l_e)
        pl_x = self.inter_bn_edge_e(pl_x)
        p_x = F.relu(p_x)
        p_e = F.relu(p_e)
        l_x = F.relu(l_x)
        l_e = F.relu(l_e)
        pl_x = F.relu(pl_x)
        p_x = F.dropout(p_x, self.dropout, training=self.training)
        p_e = F.dropout(p_e, self.dropout, training=self.training)
        l_x = F.dropout(l_x, self.dropout, training=self.training)
        l_e = F.dropout(l_e, self.dropout, training=self.training)
        pl_x = F.dropout(pl_x, self.dropout, training=self.training)
        if self.residual:
            p_x = p_x_in + p_x
            p_e = p_e_in + p_e
            l_x = l_x_in + l_x
            l_e = l_e_in + l_e
            pl_x = pl_x_in + pl_x
        batch['protein'].h = p_x
        batch['protein','p2p','protein'].h = p_e
        batch['ligand'].h = l_x
        batch[ 'ligand' , 'l2l','ligand'].h = l_e
        batch[ 'inter_edge' ].h =  pl_x
        return batch

    def inner_edge_process(self,edge_index,
                              Bx, Dx, Ex, Ce,Ax):
        i = edge_index[0]
        j = edge_index[1]
        Dx_i = Dx[i]
        Ex_j = Ex[j]
        Bx_j = Bx[j]
        index = i
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        dim_size = Bx.shape[0]
        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')
        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')
        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        x = Ax + out
        return  x, e_ij

    def inter_edge_process(self,p2e_edge_index,l2e_edge_index,
                            p_Bx,l_Bx, pl_Dx, p_Ex , l_Ex,pl_Ax):
        p_i = p2e_edge_index[1]
        p_j = p2e_edge_index[0]
        l_j = l2e_edge_index[0]
        Dx_i = pl_Dx[p_i]
        p_Ex_j = p_Ex[p_j]
        l_Ex_j = l_Ex[l_j]
        p_Bx_j = p_Bx[p_j]
        l_Bx_j = l_Bx[l_j]
        p_e_ij = Dx_i + p_Ex_j
        l_e_ij = Dx_i + l_Ex_j
        p_sigma_ij = torch.sigmoid(p_e_ij)
        l_sigma_ij = torch.sigmoid(l_e_ij)
        p_sum_sigma_x = p_sigma_ij * p_Bx_j
        l_sum_sigma_x = l_sigma_ij * l_Bx_j
        numerator_eta_xj = p_sum_sigma_x + l_sum_sigma_x
        p_sum_sigma = p_sigma_ij
        l_sum_sigma = l_sigma_ij
        denominator_eta_xj = p_sum_sigma + l_sum_sigma
        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        dst_node_num = pl_Ax.size(0)
        pad_out = scatter_add(out , p_i, dim=0, dim_size=dst_node_num)
        x = pl_Ax + pad_out
        return x
