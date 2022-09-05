import torch
from torch import nn
import dgl
from dgl.nn.pytorch import NNConv
from typing import List
from .gamma_layer import activ_coef_generates
from args import TrainArgs

device = torch.device("cuda")

class Set2Set(nn.Module):
    """
    Readout function
    https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/glob.html#Set2Set
    """
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = torch.nn.LSTM(input_size=self.output_dim, hidden_size=self.input_dim, num_layers=n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, feat=None):
        graph = graph.to(device)
        if feat is None:
            feat = graph.ndata["h"]
        # feat.shape = N*D
        # feat = self.n_feats_shuffles(graph, feat) # shuffle feats (h^T_v)
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)).to(device),
                 feat.new_zeros((self.n_layers, batch_size, self.input_dim)).to(device)) # (h_0, c_0)

            q_star = feat.new_zeros(batch_size, self.output_dim).to(device) # Initial LSTM states (B*2D)

            for i in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h) # q.shape = B*D
                q = q.view(batch_size, self.input_dim)
                e = (feat * dgl.broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True) # e.shape = N*1, where broadcast nodes has shape (N*D)

                graph.ndata['e'] = e
                alpha = dgl.softmax_nodes(graph, 'e') # alpha.shape = N*1 (weights)
                graph.ndata['r'] = feat * alpha # attention readout shape = N*D -> feat (N*D) * alpha (N*1)
                readout = dgl.sum_nodes(graph, 'r') # sum nodes for each graph = B*D
                q_star = torch.cat([q, readout], dim=-1) # B*2D

            return q_star

    def n_feats_shuffles(self, graph: dgl.DGLGraph, feats: torch.Tensor):
        num_batches = graph.batch_num_nodes()
        cursor = 0
        for nb in num_batches:
            n_feats = feats[cursor: cursor+nb]
            permuted_feats, row_prem = self.shufflerow(n_feats, 0)
            feats[cursor: cursor+nb] = permuted_feats
            cursor += nb
        return feats


    def shufflerow(self, tensor: torch.Tensor, axis):
        row_perm = torch.rand(tensor.shape[:axis + 1]).argsort(axis).to(device)  # get permutation indices
        for _ in range(tensor.ndim - axis - 1):
            row_perm.unsqueeze_(-1)
        row_perm = row_perm.repeat(*[1 for _ in range(axis + 1)],
                                   *(tensor.shape[axis + 1:]))  # reformat this for the gather operation
        return tensor.gather(axis, row_perm), row_perm


class Reshape(nn.Module):
    """Reshape tensor into desired shape."""
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: torch.Tensor):
        return x.view((x.size(0),) + self.shape)


class IgnoreLSTMHidden(nn.Module):
    """Passing LSTM output only (ignore hidden state)"""
    def __init__(self):
        super(IgnoreLSTMHidden, self).__init__()

    def forward(self, x):
        out, _ = x
        return out


class Readout(nn.Module):
    """Simple readout functions."""
    def __init__(self, readout_type="sum"):
        super(Readout, self).__init__()
        self.readout_type = readout_type

    def forward(self, g: dgl.DGLGraph, feats):
        if isinstance(feats, torch.Tensor):
            g.ndata["r"] = feats
        if self.readout_type == "mean":
            return dgl.mean_nodes(g, "r")
        else:
            return dgl.sum_nodes(g, "r")


class ActivityCoefficientLayer(nn.Module):
    """
    COSMO-SAC calculate activity coefficient from sigma profile
    """
    def __init__(self):
        super(ActivityCoefficientLayer, self).__init__()

    def activation_coef_layer(self, smiles, sigma_profiles):
        preds = []
        for smi, prof in zip(smiles, sigma_profiles):
            preds.append(activ_coef_generates(smi, prof, path="tests/data/params.csv"))
        return torch.tensor(preds).to(device)

    def forward(self, smiles, sigma_profiles):
        return self.activation_coef_layer(smiles, sigma_profiles)

def create_ffn(input_dim, output_dim, hidden_dim, n_layers=3):
    if n_layers == 1:
        return nn.Sequential(
                nn.Linear(input_dim, output_dim)
        )
    ffn = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
            ]
    if n_layers >= 2:
        for _ in range(n_layers-2):
            ffn.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ])
    ffn.extend([nn.Linear(hidden_dim, output_dim)])
    ffn = nn.Sequential(*ffn)
    return ffn


def OutputLayers(args: TrainArgs, input_dim, n_layers=3) -> List[nn.Module]:
    """
    Output layers based on hidden states of the model.
    """
    output = []
    multiclass_cnt = 0
    for dataset_type, output_dim, task in zip(args.dataset_type, args.output_dim, args.task_names):
        if dataset_type == "classification":
            n_classes = args.multiclass_num[multiclass_cnt]
            output.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim * n_classes),
                    Reshape(-1, n_classes),
                )
            )
            multiclass_cnt += 1

        elif dataset_type == "smiles":

            output.append(
                nn.Sequential(
                    nn.Linear(input_dim, args.node_hidden_dim * output_dim),
                    Reshape(output_dim, args.node_hidden_dim),
                    nn.LSTM(input_size=args.node_hidden_dim, hidden_size=args.node_hidden_dim, num_layers=1,
                            batch_first=True),
                    IgnoreLSTMHidden(),
                    nn.Tanh(),
                    nn.Linear(args.node_hidden_dim, args.multiclass_num[multiclass_cnt])
                )
            )
            multiclass_cnt += 1

        elif dataset_type == "fingerprint":
            output.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.Sigmoid()
                )
            )

        elif task == "activity_coefficients":
            output.append(ActivityCoefficientLayer())

        else:
            """Regression"""
            output.append(
                # nn.Sequential(
                #     nn.Linear(input_dim, args.node_hidden_dim),
                #     nn.ReLU(),
                #     nn.Linear(args.node_hidden_dim, args.node_hidden_dim),
                #     nn.ReLU(),
                #     nn.Linear(args.node_hidden_dim, output_dim),
                # )
                create_ffn(input_dim=input_dim, output_dim=output_dim, 
                           hidden_dim=args.node_hidden_dim, n_layers=n_layers)
            )
    return output


class MPNN(nn.Module):
    """
    According to Message Passing Neural network (NNConv) and readout by Set2Set model.
    https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/conv/nnconv.html#NNConv
    """
    def __init__(self, args: TrainArgs):
        super(MPNN, self).__init__()
        self.n_classes = args.output_dim
        self.num_step_message_passing = args.mpnn_steps
        self.encode = nn.Sequential(
            nn.Linear(args.node_feat_dim, args.node_hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
        )

        edge_net = nn.Sequential(
            nn.Linear(args.edge_feat_dim, args.edge_hidden_dim),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(args.edge_hidden_dim, args.node_hidden_dim * args.node_hidden_dim),
        )

        # Message passing layer
        self.conv = NNConv(in_feats=args.node_hidden_dim, out_feats=args.node_hidden_dim, edge_func=edge_net, aggregator_type="sum")
        self.gru = nn.GRU(args.node_hidden_dim, args.node_hidden_dim, num_layers=1)

        ## Readout functions
        if args.readout == "sum" or args.readout == "mean":
            self.readout = Readout(readout_type=args.readout)
        else:
            self.readout = Set2Set(input_dim=args.node_hidden_dim, n_iters=args.s2s_steps, n_layers=args.s2s_layer_nums)
            self.decode = nn.Sequential(
                nn.Linear(2 * args.node_hidden_dim, 4 * args.node_hidden_dim),
                nn.ReLU(),
            )
        # self.output_collections = nn.ModuleList(self.last_layer(args))
        self.output_collections = nn.ModuleList(OutputLayers(args, input_dim=args.node_hidden_dim, n_layers=3))
        self.tasks = args.task_names

    def forward(self, g: dgl.DGLGraph, smiles=None, output_hidden_states=False):
        g = g.to(device)
        n_feat = g.ndata["h"]
        e_feat = g.edata["h"]
        out = self.encode(n_feat) # node_hidden_dim # (15, 127) -> (15, 200)
        h = out.unsqueeze(0) # (1, 15, 200)
        """Message Passing"""
        for i in range(self.num_step_message_passing):
            m = self.conv(g, out, e_feat) # node_hidden_dim m: (15, 200)
            m = m.unsqueeze(0) # (1, 15, 200)
            out, h = self.gru(m, h) # out (1, 15, 200)
            out = out.squeeze(0) # out (15, 200)

        """Readout"""
        out = self.readout(g, out) # (1, 200)

        """Multitask or single: DNN"""
        predict = []
        for i, output_module in enumerate(self.output_collections):
            if self.tasks[i] == "activity_coefficients":
                sigma_profile_idx = self.tasks.index("sigma_profile")
                # Sigma profile should be prior to activity coefficients
                predict.append(output_module(smiles, predict[sigma_profile_idx]))
            else:
                predict.append(output_module(out))

        if output_hidden_states:
            return predict, out
        return predict


class FFNN(nn.Module):
    """
    A simple feed forward network.
    """

    def __init__(self, args: TrainArgs):
        super(FFNN, self).__init__()

        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.hidden_dim = args.node_hidden_dim
        self.n_layers = args.s2s_layer_nums
        self._create_ffn()

    def _create_ffn(self):
        ffn = [nn.Linear(self.input_dim, self.hidden_dim)]
        for _ in range(self.n_layers - 2):
            ffn.extend([
                nn.ReLU(),##
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ])
        ffn.extend([
            nn.Linear(self.hidden_dim, self.output_dim[0]) # assume no multitask, the output list has only 1 element.
        ])

        self.ffn = nn.Sequential(*ffn)

    def forward(self, x):
        return [self.ffn(x)]


class Set2SetNN(nn.Module):
    """
    Set2Set on graph inputs.
    """
    def __init__(self, args: TrainArgs):
        super(Set2SetNN, self).__init__()
        self.node_input_dim = args.node_feat_dim
        self.edge_input_dim = args.edge_feat_dim
        self.node_hidden_dim = 2 * self.node_input_dim
        self.edge_hidden_dim = 2 * self.edge_input_dim
        self.output_dim = args.output_dim[0]
        self.n_iters = args.s2s_steps
        self.n_layers = args.s2s_layer_nums
        self.lstm_node = torch.nn.LSTM(input_size=self.node_hidden_dim, hidden_size=self.node_input_dim, num_layers=self.n_layers)
        self.lstm_edge = torch.nn.LSTM(input_size=self.edge_hidden_dim, hidden_size=self.edge_input_dim, num_layers=self.n_layers)
        self.decode = nn.Sequential(
            nn.Linear(self.node_hidden_dim + self.edge_hidden_dim, self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 4 * self.node_hidden_dim)
        )
        self.reset_parameters()
        self.output_collections = nn.ModuleList(OutputLayers(args, input_dim=4 * self.node_hidden_dim))
        self.tasks = args.task_names

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm_node.reset_parameters()
        self.lstm_edge.reset_parameters()


    def forward(self, graph: dgl.DGLGraph, smiles=None, output_hidden_states=False):
        graph = graph.to(device)
        node_feat = graph.ndata["h"]
        edge_feat = graph.edata["h"]
        # feat.shape = N*D
        with graph.local_scope():
            batch_size = graph.batch_size

            # node set2set
            h = (node_feat.new_zeros((self.n_layers, batch_size, self.node_input_dim)).to(device),
                 node_feat.new_zeros((self.n_layers, batch_size, self.node_input_dim)).to(device))

            q_star_n = node_feat.new_zeros(batch_size, self.node_hidden_dim).to(device) # Initial LSTM states (B*2D)

            for i in range(self.n_iters):
                q, h = self.lstm_node(q_star_n.unsqueeze(0), h) # q.shape = B*D
                q = q.view(batch_size, self.node_input_dim)
                e = (node_feat * dgl.broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True) # e.shape = N*1, where broadcast nodes has shape (N*D)

                graph.ndata['e'] = e
                alpha = dgl.softmax_nodes(graph, 'e') # alpha.shape = N*1 (weights)
                graph.ndata['r'] = node_feat * alpha # attention readout shape = N*D -> feat (N*D) * alpha (N*1)
                readout = dgl.sum_nodes(graph, 'r') # sum nodes for each graph = B*D
                q_star_n = torch.cat([q, readout], dim=-1) # B*2D

            # edge set2set
            h = (edge_feat.new_zeros((self.n_layers, batch_size, self.edge_input_dim)).to(device),
                 edge_feat.new_zeros((self.n_layers, batch_size, self.edge_input_dim)).to(device))

            q_star_e = edge_feat.new_zeros(batch_size, self.edge_hidden_dim).to(device)  # Initial LSTM states (B*2D)

            for i in range(self.n_iters):
                q, h = self.lstm_edge(q_star_e.unsqueeze(0), h)  # q.shape = B*D
                q = q.view(batch_size, self.edge_input_dim)
                e = (edge_feat * dgl.broadcast_edges(graph, q)).sum(dim=-1,
                                                               keepdim=True)  # e.shape = N*1, where broadcast nodes has shape (N*D)

                graph.edata['e'] = e
                alpha = dgl.softmax_edges(graph, 'e')  # alpha.shape = N*1 (weights)
                graph.edata['r'] = edge_feat * alpha  # attention readout shape = N*D -> feat (N*D) * alpha (N*1)
                readout = dgl.sum_edges(graph, 'r')  # sum nodes for each graph = B*D
                q_star_e = torch.cat([q, readout], dim=-1)  # B*2D

            q_star = torch.cat([q_star_n, q_star_e], dim=-1)
            out = self.decode(q_star)

            """Multitask or single"""
            predict = []
            for i, output_module in enumerate(self.output_collections):
                if self.tasks[i] == "activity_coefficients":
                    sigma_profile_idx = self.tasks.index("sigma_profile")
                    # Sigma profile should be prior to activity coefficients
                    predict.append(output_module(smiles, predict[sigma_profile_idx]))
                else:
                    predict.append(output_module(out))
            if output_hidden_states:
                return predict, out
            return predict


class ReadoutNN(nn.Module):
    """A readout + FFNN model"""

    def __init__(self, args: TrainArgs):
        super(ReadoutNN, self).__init__()
        self.node_input_dim = args.node_feat_dim
        self.edge_input_dim = args.edge_feat_dim
        self.node_hidden_dim = args.node_hidden_dim
        self.edge_hidden_dim = args.edge_hidden_dim
        self.output_dim = args.output_dim[0]

        node_layers = [nn.Linear(self.node_input_dim, self.node_hidden_dim)]
        edge_layers = [nn.Linear(self.edge_input_dim, self.edge_hidden_dim)]
        for _ in range(args.s2s_steps):
            node_layers.append(nn.Linear(self.node_hidden_dim, self.node_hidden_dim))
            node_layers.append(nn.ReLU())
            edge_layers.append(nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim))
            edge_layers.append(nn.ReLU())

        self.readout_nodes = nn.Sequential(*node_layers)
        self.readout_edges = nn.Sequential(*edge_layers)

        self.decode = nn.Sequential(
            nn.Linear(self.node_hidden_dim + self.edge_hidden_dim, self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 4 * self.node_hidden_dim)
        )
        self.output_collections = nn.ModuleList(OutputLayers(args, input_dim=4 * self.node_hidden_dim))
        self.tasks = args.task_names


    def forward(self, graph: dgl.DGLGraph, smiles=None, output_hidden_states=False):
        graph = graph.to(device)

        node_feat = graph.ndata["h"]
        edge_feat = graph.edata["h"]

        q_node = self.readout_nodes(node_feat)
        q_edge = self.readout_edges(edge_feat)

        graph.ndata["r"] = q_node
        graph.edata["r"] = q_edge

        r_node = dgl.sum_nodes(graph, "r")
        r_edge = dgl.sum_edges(graph, "r")

        q = torch.cat([r_node, r_edge], dim=-1)
        out = self.decode(q)

        """Multitask or single"""
        predict = []
        for i, output_module in enumerate(self.output_collections):
            if self.tasks[i] == "activity_coefficients":
                sigma_profile_idx = self.tasks.index("sigma_profile")
                # Sigma profile should be prior to activity coefficients
                predict.append(output_module(smiles, predict[sigma_profile_idx]))
            else:
                predict.append(output_module(out))

        if output_hidden_states:
            return predict, out
        return predict

        return predict

