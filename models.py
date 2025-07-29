import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, DenseGATConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv


class GATBlock(nn.Module):
    def __init__(self, gat_layers_dim, heads=1, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(GATBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gat_layers_dim) - 1):
            conv_layer = GATConv(gat_layers_dim[i], gat_layers_dim[i + 1], heads=heads, dropout=dropout_rate)
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))

        return embeddings

class DenseGATBlock(nn.Module):
    def __init__(self, gat_layers_dim, heads=1, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGATBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gat_layers_dim) - 1):
            conv_layer = DenseGATConv(gat_layers_dim[i], gat_layers_dim[i + 1], heads=heads, dropout=dropout_rate)
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


class DenseGATModel(nn.Module):
    def __init__(self, layers_dim, heads=1, edge_dropout_rate=0.):
        super(DenseGATModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGATBlock(layers_dim, heads=heads, dropout_rate=0.1,
                                        relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings
    
class HetDTAGraph(nn.Module):
    def __init__(self, hidden_dim, out_dim, heads=1, dropout_rate=0.1):
        super(HetDTAGraph, self).__init__()
        
        # 定义异构图卷积层
        self.conv1 = HeteroConv({
            # 同类型节点间的边保留自环
            ('drug', 'interacts', 'drug'): GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout_rate),
            # 不同类型节点间的边禁用自环
            ('drug', 'binds', 'target'): GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout_rate, add_self_loops=False),
            ('target', 'bound_by', 'drug'): GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout_rate, add_self_loops=False),
            # 同类型节点间的边保留自环
            ('target', 'similar', 'target'): GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout_rate),
        })
        
        self.conv2 = HeteroConv({
            # 同类型节点间的边保留自环
            ('drug', 'interacts', 'drug'): GATConv(hidden_dim, out_dim, heads=heads, dropout=dropout_rate),
            # 不同类型节点间的边禁用自环
            ('drug', 'binds', 'target'): GATConv(hidden_dim, out_dim, heads=heads, dropout=dropout_rate, add_self_loops=False),
            ('target', 'bound_by', 'drug'): GATConv(hidden_dim, out_dim, heads=heads, dropout=dropout_rate, add_self_loops=False),
            # 同类型节点间的边保留自环
            ('target', 'similar', 'target'): GATConv(hidden_dim, out_dim, heads=heads, dropout=dropout_rate),
        })
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x_dict, edge_index_dict):
        # 第一层异构图卷积
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # 第二层异构图卷积
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.relu(x) for key, x in x_dict.items()}
        
        return x_dict

class HeteroGATModel(nn.Module):
    def __init__(self, layers_dim, heads=1, edge_dropout_rate=0.):
        super(HeteroGATModel, self).__init__()
        
        self.edge_dropout_rate = edge_dropout_rate
        self.hidden_dim = layers_dim[1]  # 中间层维度
        self.out_dim = layers_dim[-1]    # 输出层维度
        
        # 节点类型特征投影
        self.drug_proj = nn.Linear(layers_dim[0], self.hidden_dim)
        self.target_proj = nn.Linear(layers_dim[0], self.hidden_dim)
        
        # 异构图神经网络
        self.het_conv = HetDTAGraph(
            hidden_dim=self.hidden_dim, 
            out_dim=self.out_dim,
            heads=heads, 
            dropout_rate=0.1
        )
    
    def construct_hetero_data(self, graph):
        """将同构图数据转换为异构图数据"""
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        
        # # 调试信息
        # print(f"\nConstructing heterogeneous graph:")
        # print(f"Node feature matrix shape: {xs.shape}")
        # print(f"Adjacency matrix shape: {adj.shape}")
        # print(f"Number of drugs: {num_d}, Number of targets: {num_t}")
        
        # 分离药物和靶标特征
        drug_feat = xs[:num_d]
        target_feat = xs[num_d:]
        
        # 投影到隐藏维度
        drug_hidden = self.drug_proj(drug_feat)
        target_hidden = self.target_proj(target_feat)
        
        # 节点特征字典
        x_dict = {
            'drug': drug_hidden,
            'target': target_hidden
        }
        
        # 构建空的边索引字典
        edge_index_dict = {}
        
        # 构建不同类型的边索引 - 使用安全的方法处理可能的空边
        # 药物-药物边
        drug_drug_idx = torch.where(adj[:num_d, :num_d] != 0)
        if drug_drug_idx[0].shape[0] > 0:  # 确保有边存在
            drug_drug_edge = torch.stack([drug_drug_idx[0], drug_drug_idx[1]])
            edge_index_dict[('drug', 'interacts', 'drug')] = drug_drug_edge
            # print(f"Drug-Drug edges: {drug_drug_edge.shape[1]}")
        else:
            # 创建一个空的边索引，但格式正确 (2, 0)
            edge_index_dict[('drug', 'interacts', 'drug')] = torch.zeros((2, 0), dtype=torch.long, device=xs.device)
            # print("No Drug-Drug edges found")
        
        # 药物-靶标边
        drug_target_idx = torch.where(adj[:num_d, num_d:] != 0)
        if drug_target_idx[0].shape[0] > 0:
            # 调整靶标索引，使其相对于靶标特征矩阵而非整个图
            drug_idx = drug_target_idx[0]
            target_idx = drug_target_idx[1]
            drug_target_edge = torch.stack([drug_idx, target_idx])
            edge_index_dict[('drug', 'binds', 'target')] = drug_target_edge
            # print(f"Drug-Target edges: {drug_target_edge.shape[1]}")
        else:
            edge_index_dict[('drug', 'binds', 'target')] = torch.zeros((2, 0), dtype=torch.long, device=xs.device)
            # print("No Drug-Target edges found")
        
        # 靶标-药物边
        target_drug_idx = torch.where(adj[num_d:, :num_d] != 0)
        if target_drug_idx[0].shape[0] > 0:
            # 调整靶标索引，使其相对于靶标特征矩阵而非整个图
            target_idx = target_drug_idx[0]
            drug_idx = target_drug_idx[1]
            target_drug_edge = torch.stack([target_idx, drug_idx])
            edge_index_dict[('target', 'bound_by', 'drug')] = target_drug_edge
            # print(f"Target-Drug edges: {target_drug_edge.shape[1]}")
        else:
            edge_index_dict[('target', 'bound_by', 'drug')] = torch.zeros((2, 0), dtype=torch.long, device=xs.device)
            # print("No Target-Drug edges found")
        
        # 靶标-靶标边
        target_target_idx = torch.where(adj[num_d:, num_d:] != 0)
        if target_target_idx[0].shape[0] > 0:
            target_target_edge = torch.stack([target_target_idx[0], target_target_idx[1]])
            edge_index_dict[('target', 'similar', 'target')] = target_target_edge
            # print(f"Target-Target edges: {target_target_edge.shape[1]}")
        else:
            edge_index_dict[('target', 'similar', 'target')] = torch.zeros((2, 0), dtype=torch.long, device=xs.device)
            # print("No Target-Target edges found")
        
        # 应用边缘dropout (仅对存在边的情况)
        if self.training and self.edge_dropout_rate > 0:
            for edge_type, edge_index in edge_index_dict.items():
                if edge_index.shape[1] > 0:  # 确保有边存在
                    # 对于边索引，获取相应的边权重
                    if edge_type[0] == 'drug' and edge_type[2] == 'drug':
                        i, j = edge_index
                        edge_attr = adj[i, j]
                    elif edge_type[0] == 'drug' and edge_type[2] == 'target':
                        i, j = edge_index
                        edge_attr = adj[i, j + num_d]
                    elif edge_type[0] == 'target' and edge_type[2] == 'drug':
                        i, j = edge_index
                        edge_attr = adj[i + num_d, j]
                    else:  # target->target
                        i, j = edge_index
                        edge_attr = adj[i + num_d, j + num_d]
                    
                    edge_index_drop, edge_attr_drop = dropout_adj(
                        edge_index=edge_index, 
                        edge_attr=edge_attr,
                        p=self.edge_dropout_rate,
                        force_undirected=True if edge_type[0] == edge_type[2] else False,  # 只对同类型节点强制无向
                        training=self.training
                    )
                    edge_index_dict[edge_type] = edge_index_drop
        
        # 验证所有边索引都是有效的（索引不超出节点数量）
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.shape[1] > 0:
                src_type, _, dst_type = edge_type
                
                # 检查源节点索引是否有效
                if src_type == 'drug':
                    max_idx = torch.max(edge_index[0]) if edge_index.shape[1] > 0 else -1
                    assert max_idx < num_d, f"Invalid source index for {edge_type}: {max_idx} >= {num_d}"
                else:  # target
                    max_idx = torch.max(edge_index[0]) if edge_index.shape[1] > 0 else -1
                    assert max_idx < num_t, f"Invalid source index for {edge_type}: {max_idx} >= {num_t}"
                
                # 检查目标节点索引是否有效
                if dst_type == 'drug':
                    max_idx = torch.max(edge_index[1]) if edge_index.shape[1] > 0 else -1
                    assert max_idx < num_d, f"Invalid destination index for {edge_type}: {max_idx} >= {num_d}"
                else:  # target
                    max_idx = torch.max(edge_index[1]) if edge_index.shape[1] > 0 else -1
                    assert max_idx < num_t, f"Invalid destination index for {edge_type}: {max_idx} >= {num_t}"
        
        return x_dict, edge_index_dict
    
    def forward(self, graph):
        # 构建异构图数据
        x_dict, edge_index_dict = self.construct_hetero_data(graph)
        
        # 检查是否所有必要的边类型都存在，如果不存在则添加空边
        for edge_type in [
            ('drug', 'interacts', 'drug'),
            ('drug', 'binds', 'target'),
            ('target', 'bound_by', 'drug'),
            ('target', 'similar', 'target')
        ]:
            if edge_type not in edge_index_dict:
                # print(f"Warning: Edge type {edge_type} not found in edge_index_dict. Adding empty edge.")
                edge_index_dict[edge_type] = torch.zeros((2, 0), dtype=torch.long, device=graph.x.device)
        
        # 传入异构图神经网络
        try:
            out_dict = self.het_conv(x_dict, edge_index_dict)
        except Exception as e:
            print(f"Error during heterogeneous convolution: {e}")
            print(f"x_dict keys: {list(x_dict.keys())}")
            print(f"edge_index_dict keys: {list(edge_index_dict.keys())}")
            for edge_type, edge_index in edge_index_dict.items():
                print(f"Edge type {edge_type} has shape {edge_index.shape}")
                if edge_index.shape[1] > 0:
                    print(f"  Max source index: {torch.max(edge_index[0]).item()}")
                    print(f"  Max target index: {torch.max(edge_index[1]).item()}")
            # 失败时回退到同构方式处理
            all_embeddings = torch.zeros(
                (graph.num_drug + graph.num_target, self.out_dim), 
                device=graph.x.device
            )
            return [all_embeddings]
        
        # 合并结果，与原模型保持一致的输出格式
        num_d = graph.num_drug
        num_t = graph.num_target
        
        # 获取所有节点的嵌入表示
        all_embeddings = torch.zeros(
            (num_d + num_t, self.out_dim), 
            device=graph.x.device
        )
        
        # 确保'drug'和'target'键都存在于out_dict中
        if 'drug' in out_dict:
            all_embeddings[:num_d] = out_dict['drug']
        else:
            print("Warning: 'drug' key not found in out_dict. Using zeros for drug embeddings.")
        
        if 'target' in out_dict:
            all_embeddings[num_d:] = out_dict['target']
        else:
            print("Warning: 'target' key not found in out_dict. Using zeros for target embeddings.")
        
        # 模拟原模型返回最后一层嵌入
        return [all_embeddings]


class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings

class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam, target_hidden_dim=1280):
        super(Contrast, self).__init__()

        # 药物的 proj
        self.drug_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))

        # 靶标的 proj
        self.target_proj = nn.Sequential(
            nn.Linear(target_hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))

        self.tau = tau
        self.lam = lam
        for model in list(self.drug_proj.children()) + list(self.target_proj.children()):
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
    
    def enhance_embedding(self, embedding, is_target=False):
        """增强冷启动场景的嵌入表示"""
        # 使用自我投影增强泛化能力
        proj = self.target_proj if is_target else self.drug_proj
        
        # 注意：embedding可能已经是投影后的结果（维度为output_dim）
        # 或者是原始嵌入（维度为hidden_dim或target_hidden_dim）
        if embedding.size(1) == proj[0].in_features:  # 检查是否是原始维度
            # 完整投影
            enhanced = proj(embedding)
            enhanced = F.normalize(enhanced, p=2, dim=1)
            return enhanced
        elif embedding.size(1) == proj[-1].out_features:  # 已经是投影后的维度
            # 已经是正确维度，仅正则化
            return F.normalize(embedding, p=2, dim=1)
        else:
            # 维度不匹配，打印错误信息并尝试适应
            print(f"Warning: Embedding dimension mismatch in enhance_embedding. "
                  f"Got {embedding.size(1)}, expected {proj[0].in_features} or {proj[-1].out_features}. "
                  f"Returning normalized input.")
            return F.normalize(embedding, p=2, dim=1)

    def info_nce_loss(self, z1, z2, pos_mask):
        """
        InfoNCE 对比损失
        Args:
            z1: 第一组嵌入
            z2: 第二组嵌入
            pos_mask: 正样本掩码
        """
        # 归一化
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z1, z2.t()) / self.tau
        
        # 计算对比损失
        pos_sim = sim_matrix[pos_mask]
        neg_sim = sim_matrix[~pos_mask]

        # 正样本与所有样本的对比
        pos_loss = -torch.log(
            torch.exp(pos_sim) / 
            (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim)))
        ).mean()

        return pos_loss

    def forward(self, za, zb, pos, is_target=False):
        """
        对比学习前向传播
        Args:
            za: 第一组原始嵌入
            zb: 第二组原始嵌入
            pos: 正样本掩码
            is_target: 是否是靶标
        """
        # 投影
        if is_target:
            za_proj = self.target_proj(za)
            zb_proj = self.target_proj(zb)
        else:
            za_proj = self.drug_proj(za)
            zb_proj = self.drug_proj(zb)

        # 转换正样本掩码
        pos_mask_a2b = pos.to_dense().bool()
        pos_mask_b2a = pos.t().to_dense().bool()

        # 计算双向 InfoNCE 损失
        lori_a = self.info_nce_loss(za_proj, zb_proj, pos_mask_a2b)
        lori_b = self.info_nce_loss(zb_proj, za_proj, pos_mask_b2a)

        # 综合损失
        total_loss = self.lam * lori_a + (1 - self.lam) * lori_b

        return total_loss, torch.cat((za_proj, zb_proj), 1)


class UEC2DTA(nn.Module):
    def __init__(self, tau, lam, ns_dims, d_ms_dims, embedding_dim=128, heads=1, dropout_rate=0., 
                 use_unimol=False, drug_hidden_dim=None, target_hidden_dim=1280, use_hetero=True):
        super(UEC2DTA, self).__init__()

        self.output_dim = embedding_dim * 2
        self.target_hidden_dim = target_hidden_dim
        self.use_unimol = use_unimol
        self.drug_hidden_dim = drug_hidden_dim
        self.use_hetero = use_hetero  # 新增: 是否使用异构图模型

        # 使用异构图神经网络替代Dense GAT
        if use_hetero:
            self.affinity_graph_conv = HeteroGATModel(ns_dims, heads=heads, edge_dropout_rate=dropout_rate)
        else:
            self.affinity_graph_conv = DenseGATModel(ns_dims, heads=heads, edge_dropout_rate=dropout_rate)
        
        # 根据是否使用UniMol选择药物特征处理方式
        if self.use_unimol:
            # 如果使用UniMol，我们需要一个线性层来处理药物特征，以便与对比学习兼容
            self.drug_feature_proj = nn.Sequential(
                nn.Linear(drug_hidden_dim, ns_dims[-1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            # 使用传统图神经网络处理药物
            self.drug_graph_conv = GATModel(d_ms_dims, heads=heads)
        
        self.drug_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        self.target_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam, target_hidden_dim=target_hidden_dim)
        
    def enhance_cold_start_embeddings(self, drug_embeddings, target_embeddings, scenario='S1'):
        """为冷启动场景增强嵌入"""
        if scenario == 'S1':
            return drug_embeddings, target_embeddings
        
        # 针对S2场景的药物嵌入增强
        if scenario in ['S2', 'S4']:
            drug_embeddings = self.drug_contrast.enhance_embedding(drug_embeddings, is_target=False)
        
        # 针对S3场景的靶标嵌入增强
        if scenario in ['S3', 'S4']:
            target_embeddings = self.target_contrast.enhance_embedding(target_embeddings, is_target=True)
        
        return drug_embeddings, target_embeddings

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_pos, target_pos, scenario='S1', enhance_embeddings=True):
        num_d = affinity_graph.num_drug
        num_t = affinity_graph.num_target

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]
        
        # 根据是否使用UniMol处理药物特征
        if self.use_unimol:
            # UniMol特征已经是一个张量 [num_drug, drug_hidden_dim]
            # 检查特征维度并进行必要的调整
            if drug_graph_batchs.size(-1) != self.drug_hidden_dim:
                print(f"Warning: Drug feature dimension ({drug_graph_batchs.size(-1)}) doesn't match model dimension ({self.drug_hidden_dim})")
                if drug_graph_batchs.size(-1) > self.drug_hidden_dim:
                    # 如果特征维度较大，截断多余部分
                    print(f"Truncating drug features from {drug_graph_batchs.size(-1)} to {self.drug_hidden_dim}")
                    drug_graph_batchs = drug_graph_batchs[:, :self.drug_hidden_dim]
                else:
                    # 如果特征维度较小，填充零值
                    print(f"Padding drug features from {drug_graph_batchs.size(-1)} to {self.drug_hidden_dim}")
                    padding = torch.zeros((drug_graph_batchs.size(0), 
                                        self.drug_hidden_dim - drug_graph_batchs.size(-1)), 
                                        device=drug_graph_batchs.device)
                    drug_graph_batchs = torch.cat([drug_graph_batchs, padding], dim=1)
            
            drug_graph_embedding = self.drug_feature_proj(drug_graph_batchs)
        else:
            # 传统的图神经网络特征提取
            drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]

        # 获取有效的药物索引（不同场景可能不同）
        valid_drug_indices = torch.arange(num_d, device=affinity_graph.adj.device)
        if torch.sum(affinity_graph.adj[:num_d]) == 0:  # 如果是S2等场景
            valid_drug_indices = torch.where(torch.sum(affinity_graph.adj[:num_d], dim=1) != 0)[0]
        
        # 输出药物处理信息
        if not hasattr(self, 'printed_drug_info'):
            print(f"\nDrug processing information:")
            print(f"Total number of drugs: {num_d}")
            print(f"Number of valid drugs: {len(valid_drug_indices)}")
            self.printed_drug_info = True
        
        # 提取有效的药物嵌入
        valid_affinity_drug_embedding = affinity_graph_embedding[valid_drug_indices]
        valid_drug_graph_embedding = drug_graph_embedding[valid_drug_indices]

        # 计算药物对比损失
        dru_loss, valid_drug_embedding = self.drug_contrast(valid_affinity_drug_embedding, valid_drug_graph_embedding, drug_pos)
        
        # 获取有效的靶标索引
        valid_target_indices = torch.arange(num_t, device=affinity_graph.adj.device)
        if torch.sum(affinity_graph.adj[num_d:]) == 0:  # 如果是S3等场景
            valid_target_indices = torch.where(torch.sum(affinity_graph.adj[num_d:], dim=1) != 0)[0]
        
        # 输出靶标处理信息
        if not hasattr(self, 'printed_target_info'):
            print(f"\nTarget processing information:")
            print(f"Total number of targets: {num_t}")
            print(f"Number of valid targets: {len(valid_target_indices)}")
            self.printed_target_info = True
        
        # 提取有效的靶标嵌入
        target_affinity_embedding = affinity_graph_embedding[num_d:][valid_target_indices]
        valid_target_graph_batchs = target_graph_batchs[valid_target_indices]
        
        # 处理靶标维度
        if valid_target_graph_batchs.size(-1) != self.target_hidden_dim:
            if valid_target_graph_batchs.size(-1) > self.target_hidden_dim:
                valid_target_graph_batchs = valid_target_graph_batchs[:, :self.target_hidden_dim]
            else:
                padding = torch.zeros((valid_target_graph_batchs.size(0), 
                                    self.target_hidden_dim - valid_target_graph_batchs.size(-1)), 
                                    device=valid_target_graph_batchs.device)
                valid_target_graph_batchs = torch.cat([valid_target_graph_batchs, padding], dim=1)
        
        target_affinity_embedding = target_affinity_embedding.unsqueeze(1)
        current_feature_dim = target_affinity_embedding.size(2)
        expand_factor = self.target_hidden_dim // current_feature_dim
        if self.target_hidden_dim % current_feature_dim != 0:
            expand_factor = (self.target_hidden_dim + current_feature_dim - 1) // current_feature_dim
            
        target_affinity_embedding = target_affinity_embedding.expand(-1, expand_factor, -1)
        batch_size = target_affinity_embedding.size(0)
        target_affinity_embedding = target_affinity_embedding.reshape(batch_size, -1)
        
        if target_affinity_embedding.size(1) > self.target_hidden_dim:
            target_affinity_embedding = target_affinity_embedding[:, :self.target_hidden_dim]
        elif target_affinity_embedding.size(1) < self.target_hidden_dim:
            padding = torch.zeros((batch_size, self.target_hidden_dim - target_affinity_embedding.size(1)), 
                                device=target_affinity_embedding.device)
            target_affinity_embedding = torch.cat([target_affinity_embedding, padding], dim=1)

        # 计算靶标对比损失
        tar_loss, valid_target_embedding = self.target_contrast(target_affinity_embedding, valid_target_graph_batchs, target_pos, is_target=True)

        # 构建完整的嵌入
        full_drug_embedding = torch.zeros((num_d, valid_drug_embedding.size(1)), device=valid_drug_embedding.device)
        full_drug_embedding[valid_drug_indices] = valid_drug_embedding

        full_target_embedding = torch.zeros((num_t, valid_target_embedding.size(1)), device=valid_target_embedding.device)
        full_target_embedding[valid_target_indices] = valid_target_embedding
        
        # 针对冷启动场景增强嵌入
        if enhance_embeddings and scenario != 'S1':
            try:
                full_drug_embedding, full_target_embedding = self.enhance_cold_start_embeddings(
                    full_drug_embedding, full_target_embedding, scenario)
            except Exception as e:
                print(f"Warning: Failed to enhance embeddings for {scenario} scenario: {e}")
                print("Using original embeddings instead.")

        return dru_loss + tar_loss, full_drug_embedding, full_target_embedding
    

class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()

        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 4 * dim)
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y
        
        # 检查是否传入了预处理的药物嵌入
        if isinstance(drug_embedding, torch.Tensor) and drug_embedding.dim() == 2 and drug_embedding.size(0) == len(drug_id):
            # 已经是预处理过的药物嵌入，直接使用
            drug_feature = drug_embedding
        else:
            # 正常情况下根据ID获取嵌入
            drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
            
        # 检查是否传入了预处理的靶标嵌入
        if isinstance(target_embedding, torch.Tensor) and target_embedding.dim() == 2 and target_embedding.size(0) == len(target_id):
            # 已经是预处理过的靶标嵌入，直接使用
            target_feature = target_embedding
        else:
            # 正常情况下根据ID获取嵌入
            target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings

class GATModel(nn.Module):
    def __init__(self, layers_dim, heads=1):
        super(GATModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GATBlock(layers_dim, heads=heads, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        embedding_batchs = list(
            map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings
