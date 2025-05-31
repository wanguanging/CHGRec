import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import dill

from .hetero_gcn import hetero_effect_graph

from .layers import kanChebConv


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + torch.eye(adj.shape[0]).to(device))

        adj = adj.float().to(device)
        self.edge_index, self.edge_weight = torch_geometric.utils.dense_to_sparse(adj)
        self.x = torch.eye(voc_size, dtype=torch.float32).to(device)

        self.gcn1 = kanChebConv(voc_size, emb_dim, K=3)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = kanChebConv(emb_dim, emb_dim, K=3)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.edge_index, self.edge_weight)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.edge_index, self.edge_weight)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = mx.sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx


class Attention_drug(nn.Module):
    def __init__(self, embed_dim, device):
        super(Attention_drug, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.scale_factor = torch.sqrt(torch.FloatTensor([131])).to(device)

    def forward(self, query, key, value):
        Q = self.query_layer(query)
        K = self.key_layer(key)
        V = self.value_layer(value)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale_factor
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

class CausaltyReview(nn.Module):
    def __init__(self, casual_graph, num_diag, num_proc, num_med):
        super(CausaltyReview, self).__init__()

        self.num_med = num_med
        self.c1 = casual_graph
        diag_med_high = casual_graph.get_threshold_effect(0.97, "Diag", "Med")
        diag_med_low = casual_graph.get_threshold_effect(0.90, "Diag", "Med")
        proc_med_high = casual_graph.get_threshold_effect(0.97, "Proc", "Med")
        proc_med_low = casual_graph.get_threshold_effect(0.90, "Proc", "Med")
        self.c1_high_limit = nn.Parameter(torch.tensor([diag_med_high, proc_med_high]))  # 选用的97%
        self.c1_low_limit = nn.Parameter(torch.tensor([diag_med_low, proc_med_low]))  # 选用的90%
        self.c1_minus_weight = nn.Parameter(torch.tensor(0.05))
        self.c1_plus_weight = nn.Parameter(torch.tensor(0.05))

    def forward(self, pre_prob, diags, procs):
        reviewed_prob = pre_prob.clone()

        for m in range(self.num_med):
            max_cdm = 0.0
            max_cpm = 0.0
            for d in diags:
                cdm = self.c1.get_effect(d, m, "Diag", "Med")
                max_cdm = max(max_cdm, cdm)
            for p in procs:
                cpm = self.c1.get_effect(p, m, "Proc", "Med")
                max_cpm = max(max_cpm, cpm)

            if max_cdm < self.c1_low_limit[0] and max_cpm < self.c1_low_limit[1]:
                reviewed_prob[0, m] -= self.c1_minus_weight
            elif max_cdm > self.c1_high_limit[0] or max_cpm > self.c1_high_limit[1]:
                reviewed_prob[0, m] += self.c1_plus_weight

        return reviewed_prob

class Aggregation(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(Aggregation, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU()
        )

        self.gate_layer = nn.Linear(32, 1)

    def forward(self, seqs):
        gates = self.gate_layer(self.h1(seqs))
        output = F.sigmoid(gates)

        return output




class CHGRec(torch.nn.Module):
    def __init__(
            self,
            causal_graph,
            mole_relevance,
            tensor_ddi_adj,
            emb_dim,
            voc_size,
            dropout,
            ehr_adj,
            device=torch.device('cpu'),
    ):
        super(CHGRec, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.voc_size = voc_size

        # Embedding of all entities
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim),
            torch.nn.Embedding(voc_size[2], emb_dim),  # 这里不用embedding【2】
            torch.nn.Embedding(voc_size[3], emb_dim)
        ])

        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()

        self.causal_graph = causal_graph

        self.mole_med_relevance = nn.Parameter(torch.tensor(mole_relevance[2], dtype=torch.float))

        self.hetero_graph = torch.nn.ModuleList([
            hetero_effect_graph(emb_dim, emb_dim, device),
            hetero_effect_graph(emb_dim, emb_dim, device),
        ])

        # Isomeric and isomeric addition parameters
        self.rho = nn.Parameter(torch.ones(3, 2))

        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        self.lstms = torch.nn.ModuleList([
            torch.nn.LSTM(emb_dim, emb_dim, batch_first=True),
            torch.nn.LSTM(emb_dim, emb_dim, batch_first=True),
            torch.nn.LSTM(emb_dim, emb_dim, batch_first=True)
        ])

        # Convert patient information to drug score
        self.query_dia = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )

        self.query_pro = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )

        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 2, emb_dim)
        )

        self.review = CausaltyReview(self.causal_graph, voc_size[0], voc_size[1], voc_size[2])

        self.tensor_ddi_adj = tensor_ddi_adj
        self.init_weights()
        self.poly = Aggregation(emb_dim * 2)
        self.diag_adj = mole_relevance[4].to_numpy()
        self.proc_adj = mole_relevance[5].to_numpy()
        self.drug_att_dia = Attention_drug(emb_dim, device)
        self.drug_att_pro = Attention_drug(emb_dim, device)
        self.out_dia = nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), nn.ReLU())
        self.out_pro = nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), nn.ReLU())
        self.layernorm_dia = nn.LayerNorm(emb_dim)
        self.layernorm_pro = nn.LayerNorm(emb_dim)
        self.layernorm = nn.LayerNorm(emb_dim)
        self.classication = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim * 6, voc_size[2]))
        self.ehr_gcn = GCN(voc_size=voc_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=voc_size[2], emb_dim=emb_dim, adj=tensor_ddi_adj, device=device)
        self.out = nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), nn.ReLU())
        self.drug_att = Attention_drug(emb_dim, device)
        self.inter1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

    def med_embedding(self, idx, emb_mole):
        # 获取所有的相关性
        relevance = self.mole_med_relevance[idx, :].to(self.device)

        # 创建一个掩码，标识非零元素的位置
        mask = relevance != 0

        # 将零元素设置为一个很小的负数（-inf），以便在softmax中保持其值为零
        relevance_masked = relevance.masked_fill(~mask, -float('inf'))

        # 对掩码后的relevance的每一行使用softmax进行归一化
        relevance_normalized = F.softmax(relevance_masked, dim=1)

        # 使用广播和批量矩阵乘法计算嵌入
        emb_med1 = torch.matmul(relevance_normalized, emb_mole.squeeze(0))

        return emb_med1.unsqueeze(0)

    def forward(self, patient_data):
        preser = []
        seq_diag, seq_proc, seq_med = [], [], []
        for adm_id, adm in enumerate(patient_data):

            idx_diag = torch.LongTensor(adm[0]).to(self.device)
            idx_proc = torch.LongTensor(adm[1]).to(self.device)
            emb_diag = self.rnn_dropout(self.embeddings[0](idx_diag)).unsqueeze(0)
            emb_proc = self.rnn_dropout(self.embeddings[1](idx_proc)).unsqueeze(0)

            # 对上次药物包的学习
            if adm == patient_data[0]:
                emb_med = torch.zeros(1, 1, self.emb_dim).to(self.device)

            else:
                adm_last = patient_data[adm_id - 1]
                idx_med = torch.LongTensor(adm_last[2]).to(self.device)
                emb_med = self.rnn_dropout(self.embeddings[2](idx_med)).unsqueeze(0)

            seq_diag.append(torch.sum(emb_diag, keepdim=True, dim=1))
            seq_proc.append(torch.sum(emb_proc, keepdim=True, dim=1))
            seq_med.append(torch.sum(emb_med, keepdim=True, dim=1))

        seq_diag = torch.cat(seq_diag, dim=1)
        seq_proc = torch.cat(seq_proc, dim=1)
        seq_med = torch.cat(seq_med, dim=1)

        if len(patient_data) >= 2:
            patient_representation = torch.concatenate([seq_diag, seq_proc], dim=-1).squeeze(dim=0) # qt
            cur_query = patient_representation[-1:,:]
            poly_las = self.poly(cur_query)
            preser.append(len(patient_data) - 1)
            for i in range(len(patient_data)-1):
                poly_his = self.poly(patient_representation[len(patient_data)-2-i])
                s = abs(poly_las - poly_his)
                if s <= 0.05:
                    preser.append(len(patient_data)-2-i)


            preser.append(len(patient_data)-1)
            seq_diag = torch.cat([seq_diag[:, i:i+1, :] for i in reversed(preser)], dim=1)
            seq_proc = torch.cat([seq_proc[:, i:i+1, :] for i in reversed(preser)], dim=1)
            seq_med = torch.cat([seq_med[:, i:i+1, :] for i in reversed(preser)], dim=1)


        output_diag, hidden_diag = self.seq_encoders[0](seq_diag)
        output_proc, hidden_proc = self.seq_encoders[1](seq_proc)
        output_med, hidden_med = self.seq_encoders[2](seq_med)
        patient_diag = torch.cat([output_diag[:, -1].flatten(), hidden_diag.flatten()])
        patient_proc = torch.cat([output_proc[:, -1].flatten(), hidden_proc.flatten()])

        emb_diags = self.embeddings[0](torch.tensor(list(range(self.voc_size[0]))).to(self.device)).unsqueeze(0)
        emb_procs = self.embeddings[1](torch.tensor(list(range(self.voc_size[1]))).to(self.device)).unsqueeze(0)
        emb_meds = self.embeddings[2](torch.tensor(list(range(self.voc_size[2]))).to(self.device)).unsqueeze(0)

        drug_dia = self.hetero_graph[0](emb_diags, emb_meds, self.diag_adj)
        drug_pro = self.hetero_graph[1](emb_procs, emb_meds, self.proc_adj)
        query_dia = self.query_dia(patient_diag).unsqueeze(0)
        query_pro = self.query_pro(patient_proc).unsqueeze(0)

        med_result_dia = self.drug_att_dia(query_dia, drug_dia, drug_dia)
        med_result_pro = self.drug_att_pro(query_pro, drug_pro, drug_pro)

        final_representations = torch.cat([self.layernorm_dia(query_dia), med_result_dia, self.layernorm_pro(query_pro), med_result_pro, output_med[:, -1], hidden_med[0]], dim=-1)


        score = self.classication(final_representations)

        score = self.review(score, patient_data[-1][0], patient_data[-1][1])

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return score, batch_neg