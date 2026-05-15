import torch
import torch as th
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_add
from torch import nn

def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        th.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

class HeteroMultiHeadAttentionLayer(nn.Module):
    """Compute attention scores with a DGLGraph's node and edge (geometric) features."""

    def __init__(self, num_input_feats, num_output_feats,
                 num_heads, using_bias=False, update_edge_feats=True ,activ_fn = None ):
        super(HeteroMultiHeadAttentionLayer, self).__init__()

        # Declare shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats
        self.activ_fn = activ_fn

        # Define node features' query, key, and value tensors, and define edge features' projection tensors
        self.p_Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.p_K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.p_V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)

        self.l_Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.l_K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.l_V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)

        self.p_edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.l_edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)

        self.inter_p_K = nn.Linear(num_input_feats, self.num_output_feats * 2 * self.num_heads, bias=using_bias)
        self.inter_p_V = nn.Linear(num_input_feats, self.num_output_feats * 2 * self.num_heads, bias=using_bias)

        self.inter_l_K = nn.Linear(num_input_feats, self.num_output_feats * 2 * self.num_heads, bias=using_bias)
        self.inter_l_V = nn.Linear(num_input_feats, self.num_output_feats * 2 * self.num_heads, bias=using_bias)

        self.pl_edge_feats_projection_q = nn.Linear(num_input_feats * 2 , self.num_output_feats * 2 * self.num_heads, bias=using_bias)

        # self.p2e_edge_type_embedding = nn.Linear(2, self.num_output_feats * self.num_heads,bias=using_bias)
        # self.l2e_edge_type_embedding = nn.Linear(2, self.num_output_feats * self.num_heads,bias=using_bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0

        if self.using_bias:
            glorot_orthogonal(self.p_Q.weight, scale=scale)
            self.p_Q.bias.data.fill_(0)
            glorot_orthogonal(self.p_K.weight, scale=scale)
            self.p_K.bias.data.fill_(0)
            glorot_orthogonal(self.p_V.weight, scale=scale)
            self.p_V.bias.data.fill_(0)

            glorot_orthogonal(self.l_Q.weight, scale=scale)
            self.l_Q.bias.data.fill_(0)
            glorot_orthogonal(self.l_K.weight, scale=scale)
            self.l_K.bias.data.fill_(0)
            glorot_orthogonal(self.l_V.weight, scale=scale)
            self.l_V.bias.data.fill_(0)

            glorot_orthogonal(self.p_edge_feats_projection.weight, scale=scale)
            self.p_edge_feats_projection.bias.data.fill_(0)
            glorot_orthogonal(self.l_edge_feats_projection.weight, scale=scale)
            self.l_edge_feats_projection.bias.data.fill_(0)

            glorot_orthogonal(self.inter_p_K.weight, scale=scale)
            self.inter_p_K.bias.data.fill_(0)
            glorot_orthogonal(self.inter_p_V.weight, scale=scale)
            self.inter_p_V.bias.data.fill_(0)

            glorot_orthogonal(self.inter_l_K.weight, scale=scale)
            self.inter_l_K.bias.data.fill_(0)
            glorot_orthogonal(self.inter_l_V.weight, scale=scale)
            self.inter_l_V.bias.data.fill_(0)

            glorot_orthogonal(self.pl_edge_feats_projection_q.weight, scale=scale)
            self.pl_edge_feats_projection_q.bias.data.fill_(0)

        else:
            glorot_orthogonal(self.p_Q.weight, scale=scale)
            glorot_orthogonal(self.p_K.weight, scale=scale)
            glorot_orthogonal(self.p_V.weight, scale=scale)

            glorot_orthogonal(self.l_Q.weight, scale=scale)
            glorot_orthogonal(self.l_K.weight, scale=scale)
            glorot_orthogonal(self.l_V.weight, scale=scale)

            glorot_orthogonal(self.p_edge_feats_projection.weight, scale=scale)
            glorot_orthogonal(self.l_edge_feats_projection.weight, scale=scale)

            glorot_orthogonal(self.inter_p_K.weight, scale=scale)
            glorot_orthogonal(self.inter_p_V.weight, scale=scale)

            glorot_orthogonal(self.inter_l_K.weight, scale=scale)
            glorot_orthogonal(self.inter_l_V.weight, scale=scale)

            glorot_orthogonal(self.pl_edge_feats_projection_q.weight, scale=scale)

    def propagate_attention(self, g):

        def inner_edge_process(tmp_g , etype  ): # 'protein' ,'p2p' ,'protein'
            src_type = etype[0]
            dst_type = etype[2]

            src_ids = tmp_g[etype].edge_index[0]
            dst_ids = tmp_g[etype].edge_index[1]

            src_kh = tmp_g[src_type].K_h[src_ids]
            dst_qh = tmp_g[dst_type].Q_h[dst_ids]
            proj_e = tmp_g[etype].proj_e

            tmp = (src_kh * dst_qh / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0) * proj_e

            tmp_g[etype].e_out = tmp
            tmp_g[etype].score = th.exp(tmp.sum(-1, keepdim=True).clamp(-5.0, 5.0))

            src_vh = tmp_g[src_type].V_h[src_ids]
            dst_node_num = tmp_g[dst_type].feats.size(0)
            tmp_g[dst_type].wV = scatter_add(src_vh * tmp_g[etype].score  ,  dst_ids  , dim=0, dim_size=dst_node_num)
            tmp_g[dst_type].z = scatter_add( tmp_g[etype].score , dst_ids , dim=0 , dim_size= dst_node_num )

        inner_edge_process(g,('protein' ,'p2p' ,'protein'))
        inner_edge_process(g,('ligand' ,'l2l' ,'ligand'))

        def inter_edge_process(tmp_g ):

            p2e_edge_ids = tmp_g[ 'protein' ,'p2e' , 'inter_edge'].edge_index[1] # p2e edge num
            l2e_edge_ids = tmp_g['ligand'  , 'l2e'  ,'inter_edge'].edge_index[1] # p2e edge num


            p2e_edge_Q_h = tmp_g['inter_edge'].inter_proj_e_q[p2e_edge_ids] ## Q : # p2e edge num , dim
            l2e_edge_Q_h = tmp_g['inter_edge'].inter_proj_e_q[l2e_edge_ids] ## Q

            p2e_protein_ids = tmp_g[ 'protein' ,'p2e' , 'inter_edge'].edge_index[0] # p2e edge num
            l2e_ligand_ids = tmp_g['ligand'  , 'l2e'  ,'inter_edge'].edge_index[0] # p2e edge num

            p2e_protein_K_h = tmp_g['protein'].inter_K_h[p2e_protein_ids] # p2e edge num ,dim
            l2e_ligand_K_h = tmp_g['ligand'].inter_K_h[l2e_ligand_ids] # p2e edge num ,dim

            p2e_protein_V_h = tmp_g['protein'].inter_V_h[p2e_protein_ids] # p2e edge num ,dim
            l2e_ligand_V_h = tmp_g['ligand'].inter_V_h[l2e_ligand_ids] # p2e edge num ,dim

            # p2e_score_energy = (p2e_protein_K_h * p2e_edge_Q_h / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0) * g['protein' ,'p2e' , 'inter_edge'].etype_emb ## * edge feat    #   p2e edge num , dim
            # l2e_score_energy = (l2e_ligand_K_h * l2e_edge_Q_h / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0) * g['ligand' ,'l2e' , 'inter_edge'].etype_emb ##

            p2e_score_energy = (p2e_protein_K_h * p2e_edge_Q_h / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)   ## * edge feat    #   p2e edge num , dim
            l2e_score_energy = (l2e_ligand_K_h * l2e_edge_Q_h / np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)   ##

            p2e_score = th.exp(p2e_score_energy.sum(-1, keepdim=True).clamp(-5.0, 5.0)) # p2e edge num
            l2e_score = th.exp(l2e_score_energy.sum(-1, keepdim=True).clamp(-5.0, 5.0)) # l2e edge num

            dst_node_num = tmp_g['inter_edge'].feats.size(0)

            tmp_g['inter_edge'].p2e_wV = scatter_add(p2e_protein_V_h * p2e_score  ,  p2e_edge_ids  , dim=0, dim_size=dst_node_num)
            tmp_g['inter_edge'].l2e_wV = scatter_add(l2e_ligand_V_h * l2e_score, l2e_edge_ids, dim=0,dim_size=dst_node_num)
            tmp_g['inter_edge'].p2e_z = scatter_add( p2e_score , p2e_edge_ids , dim=0 , dim_size= dst_node_num )
            tmp_g['inter_edge'].l2e_z = scatter_add(l2e_score, l2e_edge_ids, dim=0, dim_size=dst_node_num)
            tmp_g['inter_edge'].inter_e_out = (tmp_g['inter_edge'].p2e_wV + tmp_g['inter_edge'].l2e_wV) / (tmp_g['inter_edge'].p2e_z + tmp_g['inter_edge'].l2e_z + th.full_like(tmp_g['inter_edge'].l2e_z, 1e-6))

        inter_edge_process(g)

    def forward(self, g, p_node_feats, p_edge_feats , l_node_feats, l_edge_feats  , pl_edge_feats   ):

        p_e_out = None
        l_e_out = None
        p_node_feats_q = self.p_Q(p_node_feats)
        p_node_feats_k = self.p_K(p_node_feats)
        p_node_feats_v = self.p_V(p_node_feats)

        l_node_feats_q = self.l_Q(l_node_feats)
        l_node_feats_k = self.l_K(l_node_feats)
        l_node_feats_v = self.l_V(l_node_feats)

        p_edge_feats_projection = self.p_edge_feats_projection(p_edge_feats)
        l_edge_feats_projection = self.l_edge_feats_projection(l_edge_feats)

        inter_p_node_feats_k = self.inter_p_K(p_node_feats)
        inter_p_node_feats_v = self.inter_p_V(p_node_feats)

        inter_l_node_feats_k = self.inter_l_K(l_node_feats)
        inter_l_node_feats_v = self.inter_l_V(l_node_feats)

        pl_edge_feats_projection_q = self.pl_edge_feats_projection_q(pl_edge_feats) # edge 作为 Q

        # p2e_edge_type_embedding = self.p2e_edge_type_embedding(g['protein' ,'p2e' , 'inter_edge'].feats.float())
        # l2e_edge_type_embedding = self.l2e_edge_type_embedding(g['ligand' , 'l2e' , 'inter_edge'].feats.float())

        # Reshape tensors into [num_nodes, num_heads, feat_dim] to get projections for multi-head attention
        g['protein'].Q_h = p_node_feats_q.view(-1, self.num_heads, self.num_output_feats)
        g['protein'].K_h = p_node_feats_k.view(-1, self.num_heads, self.num_output_feats)
        g['protein'].V_h = p_node_feats_v.view(-1, self.num_heads, self.num_output_feats)

        g['ligand'].Q_h = l_node_feats_q.view(-1, self.num_heads, self.num_output_feats)
        g['ligand'].K_h = l_node_feats_k.view(-1, self.num_heads, self.num_output_feats)
        g['ligand'].V_h = l_node_feats_v.view(-1, self.num_heads, self.num_output_feats)

        g[ 'protein', 'p2p' ,'protein' ].proj_e = p_edge_feats_projection.view(-1, self.num_heads, self.num_output_feats)
        g[ 'ligand' , 'l2l' ,'ligand' ].proj_e = l_edge_feats_projection.view(-1, self.num_heads, self.num_output_feats)

        g['protein'].inter_K_h = inter_p_node_feats_k.view(-1, self.num_heads * 2, self.num_output_feats)
        g['protein'].inter_V_h = inter_p_node_feats_v.view(-1, self.num_heads * 2 , self.num_output_feats)

        g['ligand'].inter_K_h = inter_l_node_feats_k.view(-1, self.num_heads * 2 , self.num_output_feats)
        g['ligand'].inter_V_h = inter_l_node_feats_v.view(-1, self.num_heads * 2 , self.num_output_feats)

        g[ 'inter_edge' ].inter_proj_e_q = pl_edge_feats_projection_q.view(-1, self.num_heads * 2 , self.num_output_feats)

        # g['protein' ,'p2e' , 'inter_edge'].etype_emb = p2e_edge_type_embedding.view(-1, self.num_heads, self.num_output_feats)
        # g['ligand' ,'l2e' , 'inter_edge'].etype_emb = l2e_edge_type_embedding.view(-1, self.num_heads, self.num_output_feats)

        # Disperse attention information
        self.propagate_attention(g)
        # Compute final node and edge representations after multi-head attention
        p_h_out = g['protein'].wV  / (g['protein'].z + th.full_like(g['protein'].z, 1e-6))  # Add eps to all
        l_h_out = g['ligand'].wV  / (g['ligand'].z + th.full_like(g['ligand'].z, 1e-6))

        pl_e_out = g['inter_edge'].inter_e_out

        if self.update_edge_feats:
            p_e_out = g['protein' ,'p2p' ,'protein'].e_out
            l_e_out = g[ 'ligand' ,'l2l' ,'ligand'].e_out

        return p_h_out ,p_e_out  , l_h_out, l_e_out , pl_e_out

class HeteroGraphTransformerModule(nn.Module):
    """A Graph Transformer module (equivalent to one layer of graph convolutions)."""

    def __init__(
            self,
            num_hidden_channels,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
    ):
        super(HeteroGraphTransformerModule, self).__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels

        if self.apply_layer_norm:
            self.layer_norm1_p_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_p_edge_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_l_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_l_edge_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_pl_edge_feats = nn.LayerNorm(self.num_output_feats * 2)

        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_p_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_p_edge_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_l_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_l_edge_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_pl_edge_feats = nn.BatchNorm1d(self.num_output_feats * 2)

        self.hetero_mha_module = HeteroMultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=True,
            activ_fn=activ_fn
        )

        self.O_p_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_p_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_l_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_l_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_pl_edge_feats = nn.Linear(self.num_output_feats * 2 , self.num_output_feats * 2)

        # MLP for node features
        p_dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        l_dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()

        self.p_node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            p_dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.l_node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            l_dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_p_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_p_edge_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_l_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_l_edge_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_pl_edge_feats = nn.LayerNorm(self.num_output_feats * 2)

        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_p_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_p_edge_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_l_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_l_edge_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_pl_edge_feats = nn.BatchNorm1d(self.num_output_feats * 2 )

        # MLP for edge features
        self.p_edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            p_dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.l_edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            l_dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.pl_edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats* 2 , self.num_output_feats * 4, bias=False),
            self.activ_fn,
            p_dropout,
            nn.Linear(self.num_output_feats * 4, self.num_output_feats * 2 , bias=False)
        ])


        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_p_node_feats.weight, scale=scale)
        glorot_orthogonal(self.O_p_edge_feats.weight, scale=scale)
        glorot_orthogonal(self.O_l_node_feats.weight, scale=scale)
        glorot_orthogonal(self.O_l_edge_feats.weight, scale=scale)
        glorot_orthogonal(self.O_pl_edge_feats.weight,scale=scale)

        self.O_p_node_feats.bias.data.fill_(0)
        self.O_p_edge_feats.bias.data.fill_(0)
        self.O_l_node_feats.bias.data.fill_(0)
        self.O_l_edge_feats.bias.data.fill_(0)
        self.O_pl_edge_feats.bias.data.fill_(0)


        for layer in self.p_node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.p_edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.l_node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.l_edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.pl_edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)


    def run_gt_layer(self, g, p_node_feats, p_edge_feats  ,l_node_feats, l_edge_feats ,pl_edge_feats  ):
        """Perform a forward pass of graph attention using a multi-head attention (MHA) module."""
        p_node_feats_in1 = p_node_feats  # Cache node representations for first residual connection
        p_edge_feats_in1 = p_edge_feats  # Cache edge representations for first residual connection
        l_node_feats_in1 = l_node_feats  # Cache node representations for first residual connection
        l_edge_feats_in1 = l_edge_feats  # Cache edge representations for first residual connection
        pl_edge_feats_in1 = pl_edge_feats

        if self.apply_layer_norm:
            p_node_feats = self.layer_norm1_p_node_feats(p_node_feats)
            p_edge_feats = self.layer_norm1_p_edge_feats(p_edge_feats)
            l_node_feats = self.layer_norm1_l_node_feats(l_node_feats)
            l_edge_feats = self.layer_norm1_l_edge_feats(l_edge_feats)
            pl_edge_feats = self.layer_norm1_pl_edge_feats(pl_edge_feats)

        else:  # Otherwise, default to using batch normalization
            p_node_feats = self.batch_norm1_p_node_feats(p_node_feats)
            p_edge_feats = self.batch_norm1_p_edge_feats(p_edge_feats)
            l_node_feats = self.batch_norm1_l_node_feats(l_node_feats)
            l_edge_feats = self.batch_norm1_l_edge_feats(l_edge_feats)
            pl_edge_feats = self.batch_norm1_pl_edge_feats(pl_edge_feats)

        # Get multi-head attention output using provided node and edge representations
        p_node_attn_out, p_edge_attn_out ,l_node_attn_out, l_edge_attn_out ,pl_edge_attn_out   = self.hetero_mha_module(g, p_node_feats, p_edge_feats , l_node_feats, l_edge_feats ,pl_edge_feats )

        p_node_feats = p_node_attn_out.view(-1, self.num_output_feats)
        p_edge_feats = p_edge_attn_out.view(-1, self.num_output_feats)
        l_node_feats = l_node_attn_out.view(-1, self.num_output_feats)
        l_edge_feats = l_edge_attn_out.view(-1, self.num_output_feats)
        pl_edge_feats = pl_edge_attn_out.view(-1,  self.num_output_feats * 2 )

        p_node_feats = F.dropout(p_node_feats, self.dropout_rate, training=self.training)
        p_edge_feats = F.dropout(p_edge_feats, self.dropout_rate, training=self.training)
        l_node_feats = F.dropout(l_node_feats, self.dropout_rate, training=self.training)
        l_edge_feats = F.dropout(l_edge_feats, self.dropout_rate, training=self.training)
        pl_edge_feats= F.dropout(pl_edge_feats, self.dropout_rate, training=self.training)

        p_node_feats = self.O_p_node_feats( p_node_feats )
        p_edge_feats = self.O_p_edge_feats(p_edge_feats)
        l_node_feats = self.O_l_node_feats( l_node_feats )
        l_edge_feats = self.O_l_edge_feats(l_edge_feats)
        pl_edge_feats = self.O_pl_edge_feats(pl_edge_feats)

        # Make first residual connection
        if self.residual:
            p_node_feats = p_node_feats_in1 + p_node_feats  # Make first node residual connection
            p_edge_feats = p_edge_feats_in1 + p_edge_feats  # Make first edge residual connection
            l_node_feats = l_node_feats_in1 + l_node_feats  # Make first node residual connection
            l_edge_feats = l_edge_feats_in1 + l_edge_feats  # Make first edge residual connection
            pl_edge_feats  = pl_edge_feats_in1 + pl_edge_feats

        p_node_feats_in2 = p_node_feats  # Cache node representations for second residual connection
        p_edge_feats_in2 = p_edge_feats  # Cache edge representations for second residual connection
        l_node_feats_in2 = l_node_feats  # Cache node representations for second residual connection
        l_edge_feats_in2 = l_edge_feats  # Cache edge representations for second residual connection
        pl_edge_feats_in2 = pl_edge_feats

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            p_node_feats = self.layer_norm2_p_node_feats(p_node_feats)
            p_edge_feats = self.layer_norm2_p_edge_feats(p_edge_feats)
            l_node_feats = self.layer_norm2_l_node_feats(l_node_feats)
            l_edge_feats = self.layer_norm2_l_edge_feats(l_edge_feats)
            pl_edge_feats = self.layer_norm2_pl_edge_feats(pl_edge_feats)

        else:  # Otherwise, default to using batch normalization
            p_node_feats = self.batch_norm2_p_node_feats(p_node_feats)
            p_edge_feats = self.batch_norm2_p_edge_feats(p_edge_feats)
            l_node_feats = self.batch_norm2_l_node_feats(l_node_feats)
            l_edge_feats = self.batch_norm2_l_edge_feats(l_edge_feats)
            pl_edge_feats = self.batch_norm2_pl_edge_feats(pl_edge_feats)

        # Apply MLPs for node and edge features
        for layer in self.p_node_feats_MLP:
            p_node_feats = layer(p_node_feats)
        for layer in self.p_edge_feats_MLP:
            p_edge_feats = layer(p_edge_feats)
        for layer in self.l_node_feats_MLP:
            l_node_feats = layer(l_node_feats)
        for layer in self.l_edge_feats_MLP:
            l_edge_feats = layer(l_edge_feats)
        for layer in self.pl_edge_feats_MLP:
            pl_edge_feats = layer(pl_edge_feats)

        # Make second residual connection
        if self.residual:
            p_node_feats = p_node_feats_in2 + p_node_feats  # Make second node residual connection
            p_edge_feats = p_edge_feats_in2 + p_edge_feats  # Make second edge residual connection
            l_node_feats = l_node_feats_in2 + l_node_feats  # Make second node residual connection
            l_edge_feats = l_edge_feats_in2 + l_edge_feats  # Make second edge residual connection
            pl_edge_feats = pl_edge_feats_in2 + pl_edge_feats

        # Return edge representations along with node representations (for tasks other than interface prediction)
        return p_node_feats, p_edge_feats ,l_node_feats, l_edge_feats , pl_edge_feats

    def forward(self, g, p_node_feats, p_edge_feats  ,l_node_feats, l_edge_feats , pl_edge_feats  ):
        """Perform a forward pass of a Graph Transformer to get intermediate node and edge representations."""
        p_node_feats, p_edge_feats  ,l_node_feats, l_edge_feats , pl_edge_feats   = self.run_gt_layer(g, p_node_feats, p_edge_feats  ,l_node_feats, l_edge_feats , pl_edge_feats  )
        return p_node_feats, p_edge_feats  ,l_node_feats, l_edge_feats  , pl_edge_feats

class HeteroFinalGraphTransformerModule(nn.Module):
    """A (final layer) Graph Transformer module that combines node and edge representations using self-attention."""

    def __init__(self,
                 num_hidden_channels,
                 activ_fn=nn.SiLU(),
                 residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4):
        super(HeteroFinalGraphTransformerModule, self).__init__()

        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_p_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_p_edge_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_l_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_l_edge_feats = nn.LayerNorm(self.num_output_feats)

            self.layer_norm1_pl_edge_feats = nn.LayerNorm(self.num_output_feats * 2)


        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_p_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_p_edge_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_l_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_l_edge_feats = nn.BatchNorm1d(self.num_output_feats)

            self.batch_norm1_pl_edge_feats = nn.BatchNorm1d(self.num_output_feats * 2 )


        self.hetero_mha_module = HeteroMultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,  # Only use bias if a Linear() has to change sizes
            update_edge_feats=False,
            activ_fn=activ_fn
        )

        self.O_p_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_l_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        self.O_pl_edge_feats = nn.Linear(self.num_output_feats* 2 , self.num_output_feats* 2 )

        # MLP for node features
        p_dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        l_dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()

        self.p_node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            p_dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.l_node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            l_dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])

        self.pl_edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats* 2 , self.num_output_feats * 4, bias=False),
            self.activ_fn,
            p_dropout,
            nn.Linear(self.num_output_feats * 4, self.num_output_feats * 2 , bias=False)
        ])


        if self.apply_layer_norm:
            self.layer_norm2_p_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_l_node_feats = nn.LayerNorm(self.num_output_feats)

            self.layer_norm2_pl_edge_feats = nn.LayerNorm(self.num_output_feats * 2 )


        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_p_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_l_node_feats = nn.BatchNorm1d(self.num_output_feats)

            self.batch_norm2_pl_edge_feats = nn.BatchNorm1d(self.num_output_feats  * 2 )

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_p_node_feats.weight, scale=scale)
        glorot_orthogonal(self.O_l_node_feats.weight, scale=scale)
        self.O_p_node_feats.bias.data.fill_(0)
        self.O_l_node_feats.bias.data.fill_(0)

        glorot_orthogonal(self.O_pl_edge_feats.weight, scale=scale)
        self.O_pl_edge_feats.bias.data.fill_(0)

        for layer in self.p_node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)
        for layer in self.l_node_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.pl_edge_feats_MLP:
            if hasattr(layer, 'weight'):  # Skip initialization for activation functions
                glorot_orthogonal(layer.weight, scale=scale)


    def run_gt_layer(self, g, p_node_feats, p_edge_feats  , l_node_feats, l_edge_feats  ,pl_edge_feats   ):
        """Perform a forward pass of graph attention using a multi-head attention (MHA) module."""
        p_node_feats_in1 = p_node_feats  # Cache node representations for first residual connection
        l_node_feats_in1 = l_node_feats
        pl_edge_feats_in1 = pl_edge_feats

        # Apply first round of normalization before applying graph attention, for performance enhancement
        if self.apply_layer_norm:
            p_node_feats = self.layer_norm1_p_node_feats(p_node_feats)
            p_edge_feats = self.layer_norm1_p_edge_feats(p_edge_feats)
            l_node_feats = self.layer_norm1_l_node_feats(l_node_feats)
            l_edge_feats = self.layer_norm1_l_edge_feats(l_edge_feats)
            pl_edge_feats = self.layer_norm1_pl_edge_feats(pl_edge_feats)

        else:  # Otherwise, default to using batch normalization
            p_node_feats = self.batch_norm1_p_node_feats(p_node_feats)
            p_edge_feats = self.batch_norm1_p_edge_feats(p_edge_feats)
            l_node_feats = self.batch_norm1_l_node_feats(l_node_feats)
            l_edge_feats = self.batch_norm1_l_edge_feats(l_edge_feats)
            pl_edge_feats = self.batch_norm1_pl_edge_feats(pl_edge_feats)

        # Get multi-head attention output using provided node and edge representations
        p_node_attn_out, _ , l_node_attn_out ,_  , pl_edge_attn_out   = self.hetero_mha_module(g, p_node_feats, p_edge_feats , l_node_feats, l_edge_feats , pl_edge_feats )
        p_node_feats = p_node_attn_out.view(-1, self.num_output_feats)
        l_node_feats = l_node_attn_out.view(-1, self.num_output_feats)
        pl_edge_feats = pl_edge_attn_out.view(-1,  self.num_output_feats * 2 )

        p_node_feats = F.dropout(p_node_feats, self.dropout_rate, training=self.training)
        l_node_feats = F.dropout(l_node_feats, self.dropout_rate, training=self.training)
        pl_edge_feats = F.dropout(pl_edge_feats, self.dropout_rate, training=self.training)

        p_node_feats = self.O_p_node_feats(p_node_feats)
        l_node_feats = self.O_l_node_feats(l_node_feats)
        pl_edge_feats = self.O_pl_edge_feats(pl_edge_feats)

        # Make first residual connection
        if self.residual:
            p_node_feats = p_node_feats_in1 + p_node_feats  # Make first node residual connection
            l_node_feats = l_node_feats_in1 + l_node_feats  # Make first node residual connection
            pl_edge_feats = pl_edge_feats_in1 + pl_edge_feats

        p_node_feats_in2 = p_node_feats  # Cache node representations for second residual connection
        l_node_feats_in2 = l_node_feats  # Cache node representations for second residual connection
        pl_edge_feats_in2 = pl_edge_feats

        # Apply second round of normalization after first residual connection has been made
        if self.apply_layer_norm:
            p_node_feats = self.layer_norm2_p_node_feats(p_node_feats)
            l_node_feats = self.layer_norm2_l_node_feats(l_node_feats)
            pl_edge_feats = self.layer_norm2_pl_edge_feats(pl_edge_feats)
        else:  # Otherwise, default to using batch normalization
            p_node_feats = self.batch_norm2_p_node_feats(p_node_feats)
            l_node_feats = self.batch_norm2_l_node_feats(l_node_feats)
            pl_edge_feats = self.batch_norm2_pl_edge_feats(pl_edge_feats)

        # Apply MLP for node features
        for layer in self.p_node_feats_MLP:
            p_node_feats = layer(p_node_feats)
        # Apply MLP for node features
        for layer in self.l_node_feats_MLP:
            l_node_feats = layer(l_node_feats)

        for layer in self.pl_edge_feats_MLP:
            pl_edge_feats = layer(pl_edge_feats)


        # Make second residual connection
        if self.residual:
            p_node_feats = p_node_feats_in2 + p_node_feats  # Make second node residual connection
            l_node_feats = l_node_feats_in2 + l_node_feats  # Make second node residual connection
            pl_edge_feats = pl_edge_feats_in2 + pl_edge_feats

        # Return node representations
        return p_node_feats , l_node_feats ,pl_edge_feats

    def forward(self, g, p_node_feats, p_edge_feats , l_node_feats, l_edge_feats , pl_edge_feats  ):
        """Perform a forward pass of a Graph Transformer to get final node representations."""
        p_node_feats , l_node_feats ,pl_edge_feats = self.run_gt_layer(g, p_node_feats, p_edge_feats  , l_node_feats, l_edge_feats , pl_edge_feats )
        return p_node_feats , l_node_feats ,pl_edge_feats

class HeteroGraphTransformer(nn.Module):
    def __init__(
            self,
            p_in_channels=41,
            p_edge_features=10,
            l_in_channels=41,
            l_edge_features=10,
            num_hidden_channels=128,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
            **kwargs
    ):
        super(HeteroGraphTransformer, self).__init__()

        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        # --------------------
        # Initializer Modules
        # --------------------
        # Define all modules related to edge and node initialization
        self.p_node_encoder = nn.Linear(p_in_channels, num_hidden_channels)
        self.p_edge_encoder = nn.Linear(p_edge_features, num_hidden_channels)

        self.l_node_encoder = nn.Linear(l_in_channels, num_hidden_channels)
        self.l_edge_encoder = nn.Linear(l_edge_features, num_hidden_channels)

        self.pl_edge_encoder = nn.Linear(p_in_channels+l_in_channels, num_hidden_channels  * 2 )

        # --------------------
        # Transformer Module
        # --------------------
        # Define all modules related to a variable number of Graph Transformer modules
        num_intermediate_layers = max(0, num_layers - 1)

        gt_block_modules = [HeteroGraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([HeteroFinalGraphTransformerModule(
                num_hidden_channels=num_hidden_channels,
                activ_fn=activ_fn,
                residual=transformer_residual,
                num_attention_heads=num_attention_heads,
                norm_to_apply=norm_to_apply,
                dropout_rate=dropout_rate,
                num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

    def forward(self, g , p_node_feats , p_edge_feats  , l_node_feats , l_edge_feats , pl_edge_feats  ):

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



        for gt_layer in self.gt_block[:-1]:
            p_node_feats, p_edge_feats , l_node_feats , l_edge_feats , pl_edge_feats  = gt_layer(g, p_node_feats, p_edge_feats , l_node_feats , l_edge_feats  , pl_edge_feats)

        # Apply final layer to update node representations by merging current node and edge representations
        p_node_feats ,l_node_feats , pl_edge_feats  = self.gt_block[-1](g, p_node_feats, p_edge_feats , l_node_feats , l_edge_feats , pl_edge_feats )
        return p_node_feats ,l_node_feats , pl_edge_feats


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
    src_type = etype[0]     # 边的源节点类型
    src_batch = batch[src_type].batch  # 源节点的批次信息
    src_indices = batch[etype].edge_index[0]  # 边对应的源节点索引
    edge_batch = src_batch[src_indices]
    counts = torch.bincount(edge_batch, minlength=batch_size).to(device)

    return counts

def create_inter_edge( B , N_p , N_l ,  e_pl  , b_hetero_g, dtype):
    device = e_pl.device

    # torch.set_printoptions(threshold=float('inf'))
    # 构建最后要生成的完整的edge 特征
    inter_edge = th.zeros(size=(B*N_p*N_l,e_pl.size(-1) ), dtype=dtype).to(device)
    # 有用的edge特征
    feat = e_pl # edge num , dim

    batch = th.cat([th.full((1, x.type(th.int)), y) for x, y in zip(  get_batch_num_nodes(b_hetero_g , ntype='inter_edge' , device=device) , range( B ))], dim=1).reshape(-1).type(th.long).to(device)

    p_idx = b_hetero_g['inter_edge'].src_dst[:,0]
    l_idx = b_hetero_g['inter_edge'].src_dst[:,1]

    main_idx=batch * N_p * N_l + p_idx * N_l + l_idx
    inter_edge[main_idx] = feat

    inter_edge = inter_edge.view( B , N_p , N_l , -1)
    return inter_edge

class InterFocusGT(nn.Module):
    def __init__(self, hetero_model, in_channels, hidden_dim, n_gaussians, dropout_rate=0.15,
                    dist_threhold=1000):
        super(InterFocusGT, self).__init__()
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
        dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y**2,axis=-1).unsqueeze(1) + th.sum(X**2, axis=-1).unsqueeze(-1)
        return th.nan_to_num((dists**0.5).view(self.B, self.N_l,-1,atom_num_per_res),10000).min(axis=-1)[0]

