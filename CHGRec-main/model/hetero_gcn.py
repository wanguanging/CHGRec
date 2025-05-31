import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv


class hetero_effect_graph(nn.Module):
    def __init__(self, in_channels, out_channels, device, levels=5):
        super(hetero_effect_graph, self).__init__()
        self.device = device
        self.levels = levels + 1
        self.edge_type_mapping = {}
        self.initialize_edge_type_mapping()
        self.conv1 = RGCNConv(in_channels, out_channels, self.levels)
        self.conv2 = RGCNConv(out_channels, out_channels, self.levels)
    def initialize_edge_type_mapping(self):
        # 分配整数值给每种边类型
        j = 0
        for i in range(self.levels + 1):
            edge_type = ('Entity', f'connected__{i}', 'Med')
            self.edge_type_mapping[edge_type] = j
            j += 1
    def create_hetero_graph(self, emb_entity, emb_med, entity_med_weight):
        data = HeteroData()
        data['Entity'].x = emb_entity.squeeze(0)
        data['Med'].x = emb_med.squeeze(0)
        for i in range(1, self.levels):
            mask = (entity_med_weight > (i / self.levels)) & \
                   (entity_med_weight <= ((i + 1) / self.levels))
            edge_index = torch.from_numpy(np.vstack(mask.nonzero()))

            if edge_index.size(0) > 0:
                data['Entity', f'connected__{i}', 'Med'].edge_index = edge_index
        return data
    def hetero_to_homo(self, data):
        edge_index_list = []
        edge_type_list = []
        x = data['Entity'].x
        for i in range(self.levels):
            key = ('Entity', f'connected__{i}', 'Med')
            if key in data.edge_types:
                src, dst = data[key].edge_index
                edge_index_list.append(torch.stack([src, dst], dim=0))
                edge_type_list.append(torch.full((len(src),), self.edge_type_mapping[key]))

        edge_index = torch.cat(edge_index_list, dim=1).to(self.device)

        edge_type = torch.cat(edge_type_list, dim=0).to(self.device)

        return x, edge_index, edge_type

    def forward(self, emb_entity, emb_mole, entity_mole_weights):
        data = self.create_hetero_graph(emb_entity, emb_mole, entity_mole_weights)
        x, edge_index, edge_type = self.hetero_to_homo(data)
        out = self.conv1(x, edge_index, edge_type)
        out = F.relu(out)
        out = self.conv2(out, edge_index, edge_type)

        return out


