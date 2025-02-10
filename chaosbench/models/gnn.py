import torch
import torch.nn as nn
import torch.nn.functional as F


class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()


    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat

class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0,pred_len=2, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.hidden_nf = hidden_nf
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.norm = nn.LayerNorm(hidden_nf)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        out = self.norm(out)
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        lat_agg = agg.reshape(121, 240, self.hidden_nf)
        lat_agg = torch.mean(lat_agg, dim=1, keepdim=True)
        lat_agg = lat_agg.repeat(1, 240, 1).reshape(agg.size(0), self.hidden_nf)
        out = torch.cat([h, agg, lat_agg], dim=1)
        out = self.node_mlp(out) 
        if self.recurrent:
            out = out + h
            #out = self.gru(out, h)
        return out

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf,output_dim,pred_len=2,edges_in_nf=0, act_fn=nn.SiLU(), n_layers=5, attention=0, recurrent=False):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.norm = nn.LayerNorm(hidden_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=edges_in_nf,
                                              pred_len=pred_len,act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf, output_dim * self.pred_len))
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))

    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        for i in range(0, self.n_layers):
            # h = self.norm(h)
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        h = self.decoder(h)
        h = h.reshape(h.shape[0], self.pred_len, self.output_dim)
        return h
    

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result