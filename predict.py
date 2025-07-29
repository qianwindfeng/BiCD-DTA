import os
import argparse
import torch
import json
import warnings
import time  
from collections import OrderedDict
from torch import nn
from itertools import chain
from preprocess import load_data, process_data, get_target_molecule_graph, get_drug_unimol_features
from metrics import GraphDataset, collate, model_evaluate
from models import UEC2DTA, PredictModule
import numpy as np
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--scenario', type=str, default='S1', choices=['S1', 'S2', 'S3', 'S4'],
                    help='Experimental scenario (S1: random entries, S2: unseen drugs, S3: unseen targets, S4: All unseen)')
parser.add_argument('--dataset', type=str, default='davis')  #davis kiba
parser.add_argument('--epochs', type=int, default=10000)    
parser.add_argument('--batch_size', type=int, default=128)  
parser.add_argument('--lr', type=float, default=0.0002)   
parser.add_argument('--edge_dropout_rate', type=float, default=0.2)  
parser.add_argument('--tau', type=float, default=0.8)     
parser.add_argument('--lam', type=float, default=0.5,           
                   help='Balance parameter for contrastive learning')
parser.add_argument('--num_pos', type=int, default=5)     # --davis 3 kiba 5
parser.add_argument('--pos_threshold', type=float, default=8.0) # 8.0
parser.add_argument('--esm_model', type=str, default='esm2_t33_650M_UR50D', 
                    choices=['esm2_t6_8M_UR50D',
                             'esm2_t12_35M_UR50D',
                             'esm2_t33_650M_UR50D',
                             'esm2_t36_3B_UR50D', 
                             'esm2_t30_150M_UR50D'],
                    help='ESM2 model version to use')
parser.add_argument('--unimol_model', type=str, default='unimolv1', 
                    choices=['unimolv1', 
                             'unimolv2_84m', 
                             'unimolv2_164m', 
                             'unimolv2_310m', 
                             'unimolv2_570m', 
                             'unimolv2_1.1B'],
                    help='UniMol model version to use for drug feature extraction')
parser.add_argument('--unimol_batch_size', type=int, default=64, help='Batch size for UniMol processing')
parser.add_argument('--unimol_switch', type=int, default=1, choices=[0, 1],
                    help='Whether to use UniMol for drug features extraction: 0-No, 1-Yes')
parser.add_argument('--use_unimol', action='store_true', help='Whether to use UniMol for drug features extraction')


parser.add_argument('--use_hetero', action='store_true', default=True, 
                   help='Whether to use heterogeneous graph neural network instead of homogeneous GAT')
parser.add_argument('--fallback_to_homogeneous', action='store_true', 
                   help='Fallback to homogeneous model if heterogeneous model fails')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                    help='train: train model, test: load best model and evaluate')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to the model checkpoint to load')
parser.add_argument('--drug_sim_k', type=int, default=10, 
                    help='Number of similar drugs to consider in cold-start domain adaptation')
parser.add_argument('--target_sim_k', type=int, default=10, 
                    help='Number of similar targets to consider in cold-start domain adaptation')
parser.add_argument('--cosine_annealing_switch', type=int, default=1, choices=[0, 1],
                    help='Whether to use cosine annealing: 0-No, 1-Yes')            
parser.add_argument('--use_cosine_annealing', action='store_true', 
                    help='Deprecated: use --cosine_annealing_switch instead')
parser.add_argument('--lr_min', type=float, default=1e-6,  # 降低最小学习率
                    help='Minimum learning rate for cosine annealing')
parser.add_argument('--lr_max', type=float, default=0.0003,  # 降低最大学习率
                    help='Maximum learning rate for cosine annealing')
parser.add_argument('--cycle_epochs', type=int, default=50,  # 缩短周期
                    help='Number of epochs in each cosine annealing cycle')
parser.add_argument('--warmup_epochs', type=int, default=10,  # 缩短预热期
                    help='Number of epochs for warmup')
parser.add_argument('--eval_freq', type=int, default=1, 
                    help='Frequency of model evaluation (in epochs), default to evaluate every epoch')
parser.add_argument('--cold_start_adapt_epochs', type=int, default=10, 
                    help='Number of epochs to perform cold-start domain adaptation training')
parser.add_argument('--weight_alpha', type=float, default=0.5, 
                    help='Weight alpha for similarity calculation in cold-start domain adaptation')
parser.add_argument('--boost_factor', type=float, default=1.5, 
                    help='Boost factor for top similar entities (drugs/targets) in cold-start domain adaptation')
parser.add_argument('--enable_embedding_enhancement', action='store_true',
                    help='Enable embedding enhancement during cold-start adaptation (disabled by default for stability)')
parser.add_argument('--fold', type=int, default=-100, help='Fold number for cross validation. -100 means using test set')
args, _ = parser.parse_known_args()

# ESM2 model dimensions
ESM_DIMS = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t12_35M_UR50D': 480,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t33_650M_UR50D': 1280,
    'esm2_t36_3B_UR50D': 2560,
}

# UniMol model dimensions
UNIMOL_DIMS = {
    'unimolv1': 512,
    'unimolv2_84m': 768,
    'unimolv2_164m': 768,
    'unimolv2_310m': 1024,
    'unimolv2_570m': 1536,
    'unimolv2_1.1B': 1536,
}

def load_best_model(model_path, model, predictor, device):
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model performance:")
    print(f"MSE: {checkpoint['mse']:.4f}")
    print(f"RM2: {checkpoint['rm2']:.4f}")
    print(f"CI: {checkpoint['ci']:.4f}")
    print(f"Pearson: {checkpoint['pearson']:.4f}")
    
    return model, predictor, checkpoint

def train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, lr, epoch,
          batch_size, affinity_graph, drug_pos, target_pos, optimizer=None, scheduler=None):
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=0)
    
    if isinstance(drug_graphs_DataLoader, torch.Tensor):
        drug_graph_batchs = drug_graphs_DataLoader
    else:
        drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        ssl_loss, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs,
                                                                target_graphs_DataLoader, drug_pos, target_pos)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}, Current learning rate: {current_lr:.8f}")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def test(model, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos,
         target_pos, scenario='S1', drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None,
         weight_alpha=2.0, boost_factor=1.5, enhance_embeddings=False):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    
    if isinstance(drug_graphs_DataLoader, torch.Tensor):
        drug_graph_batchs = drug_graphs_DataLoader
    else:
        drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    
    attention_layer = None
    if scenario == 'S3' or scenario == 'S4':
        try:
            if isinstance(drug_graphs_DataLoader, torch.Tensor):
                drug_graph_batchs = drug_graphs_DataLoader.to(device)
            else:
                drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
            with torch.no_grad():
                _, temp_drug_embedding, temp_target_embedding = model(
                    affinity_graph.to(device), 
                    drug_graph_batchs, 
                    target_graphs_DataLoader, 
                    drug_pos, 
                    target_pos,
                    scenario,  
                    enhance_embeddings=False
                )
            target_emb_dim = temp_target_embedding.size(1)
            print(f"Creating attention layer for targets with actual embedding dimension: {target_emb_dim}")
            attention_layer = nn.Sequential(
                nn.Linear(target_emb_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(device)
        except Exception as e:
            print(f"Warning: Failed to create attention layer: {e}")
            print("Will use weighted average without attention mechanism")
    
    with torch.no_grad():
        _, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_graph_batchs, 
                                                   target_graphs_DataLoader, drug_pos, target_pos, scenario,
                                                   enhance_embeddings=enhance_embeddings)
        for data in loader:
            data = data.to(device)
            if scenario == 'S2' and drug_map is not None and drug_map_weight is not None:
                batch_size = len(data.drug_id)
                drug_id_list = data.drug_id.cpu().numpy()
                similar_drug_embeddings = torch.zeros((batch_size, drug_embedding.size(1)), device=device)
                
                for i, drug_id in enumerate(drug_id_list):
                    similar_indices = drug_map[drug_id]
                    similar_weights = drug_map_weight[drug_id].squeeze(-1)
                    valid_mask = similar_indices >= 0
                    if torch.sum(valid_mask) == 0:
                        continue
                    valid_indices = similar_indices[valid_mask]
                    valid_weights = similar_weights[valid_mask]
                    valid_weights = (valid_weights ** weight_alpha)
                    valid_weights = valid_weights / valid_weights.sum()  
                    if len(valid_weights) > 0:
                        top_idx = torch.argmax(valid_weights)
                        valid_weights[top_idx] *= boost_factor
                        valid_weights = valid_weights / valid_weights.sum() 
                    similar_embs = drug_embedding[valid_indices]
                    if attention_layer is not None and len(valid_indices) > 1:
                        try:
                            attention_scores = attention_layer(similar_embs).squeeze(-1)
                            attention_weights = F.softmax(attention_scores, dim=0)
                            combined_weights = (valid_weights * attention_weights)
                            combined_weights = combined_weights / combined_weights.sum()
                            similar_drug_embeddings[i] = torch.sum(similar_embs * combined_weights.unsqueeze(1), dim=0)
                        except Exception as e:
                            print(f"Warning: Error in attention mechanism for drug {drug_id}: {e}")
                            print(f"Falling back to weighted average without attention")
                            similar_drug_embeddings[i] = torch.sum(similar_embs * valid_weights.unsqueeze(1), dim=0)
                    else:
                        similar_drug_embeddings[i] = torch.sum(similar_embs * valid_weights.unsqueeze(1), dim=0)
                output, _ = predictor(data, similar_drug_embeddings, target_embedding)   
            elif scenario == 'S3' and target_map is not None and target_map_weight is not None:
                batch_size = len(data.target_id)
                target_id_list = data.target_id.cpu().numpy()
                similar_target_embeddings = torch.zeros((batch_size, target_embedding.size(1)), device=device)
                for i, target_id in enumerate(target_id_list):
                    similar_indices = target_map[target_id]
                    similar_weights = target_map_weight[target_id].squeeze(-1)
                    valid_mask = similar_indices >= 0
                    if torch.sum(valid_mask) == 0:
                        continue
                    valid_indices = similar_indices[valid_mask]
                    valid_weights = similar_weights[valid_mask]
                    valid_weights = (valid_weights ** weight_alpha)
                    valid_weights = valid_weights / valid_weights.sum() 
                    similar_embs = target_embedding[valid_indices]
                    if attention_layer is not None and len(valid_indices) > 1:
                        try:
                            attention_scores = attention_layer(similar_embs).squeeze(-1)
                            attention_weights = F.softmax(attention_scores, dim=0)
                            combined_weights = (valid_weights * attention_weights)
                            combined_weights = combined_weights / combined_weights.sum()
                            similar_target_embeddings[i] = torch.sum(similar_embs * combined_weights.unsqueeze(1), dim=0)
                        except Exception as e:
                            print(f"Warning: Error in attention mechanism for target {target_id}: {e}")
                            print(f"Falling back to weighted average without attention")
                            similar_target_embeddings[i] = torch.sum(similar_embs * valid_weights.unsqueeze(1), dim=0)
                    else:
                        similar_target_embeddings[i] = torch.sum(similar_embs * valid_weights.unsqueeze(1), dim=0)
                output, _ = predictor(data, drug_embedding, similar_target_embeddings)
            elif scenario == 'S4' and drug_map is not None and drug_map_weight is not None and target_map is not None and target_map_weight is not None:
                batch_size = len(data.drug_id)
                drug_id_list = data.drug_id.cpu().numpy()
                target_id_list = data.target_id.cpu().numpy()
                similar_drug_embeddings = torch.zeros((batch_size, drug_embedding.size(1)), device=device)
                for i, drug_id in enumerate(drug_id_list):
                    similar_indices = drug_map[drug_id]
                    similar_weights = drug_map_weight[drug_id].squeeze(-1)
                    valid_mask = similar_indices >= 0
                    if torch.sum(valid_mask) == 0:
                        continue
                    valid_indices = similar_indices[valid_mask]
                    valid_weights = similar_weights[valid_mask]
                    valid_weights = (valid_weights ** weight_alpha)
                    valid_weights = valid_weights / valid_weights.sum()
                    if len(valid_weights) > 0:
                        top_idx = torch.argmax(valid_weights)
                        valid_weights[top_idx] *= boost_factor
                        valid_weights = valid_weights / valid_weights.sum()
                    similar_embs = drug_embedding[valid_indices]
                    if attention_layer is not None and len(valid_indices) > 1:
                        try:
                            attention_scores = attention_layer(similar_embs).squeeze(-1)
                            attention_weights = F.softmax(attention_scores, dim=0)
                            combined_weights = (valid_weights * attention_weights)
                            combined_weights = combined_weights / combined_weights.sum()
                            similar_drug_embeddings[i] = torch.sum(similar_embs * combined_weights.unsqueeze(1), dim=0)
                        except Exception as e:
                            print(f"Warning: Error in attention mechanism for drug {drug_id} in S4: {e}")
                            print(f"Falling back to weighted average without attention")
                            similar_drug_embeddings[i] = torch.sum(similar_embs * valid_weights.unsqueeze(1), dim=0)
                    else:
                        similar_drug_embeddings[i] = torch.sum(similar_embs * valid_weights.unsqueeze(1), dim=0)
                similar_target_embeddings = torch.zeros((batch_size, target_embedding.size(1)), device=device)
                for i, target_id in enumerate(target_id_list):
                    similar_indices = target_map[target_id]
                    similar_weights = target_map_weight[target_id].squeeze(-1)
                    valid_mask = similar_indices >= 0
                    if torch.sum(valid_mask) == 0:
                        continue
                    valid_indices = similar_indices[valid_mask]
                    valid_weights = similar_weights[valid_mask]
                    valid_weights = (valid_weights ** weight_alpha)
                    valid_weights = valid_weights / valid_weights.sum()
                    similar_embs = target_embedding[valid_indices]
                    if attention_layer is not None and len(valid_indices) > 1:
                        try:
                            attention_scores = attention_layer(similar_embs).squeeze(-1)
                            attention_weights = F.softmax(attention_scores, dim=0)
                            combined_weights = (valid_weights * attention_weights)
                            combined_weights = combined_weights / combined_weights.sum()
                            similar_target_embeddings[i] = torch.sum(similar_embs * combined_weights.unsqueeze(1), dim=0)
                        except Exception as e:
                            print(f"Warning: Error in attention mechanism for target {target_id} in S4: {e}")
                            print(f"Falling back to weighted average without attention")
                            similar_target_embeddings[i] = torch.sum(similar_embs * valid_weights.unsqueeze(1), dim=0)
                    else:
                        similar_target_embeddings[i] = torch.sum(similar_embs * valid_weights.unsqueeze(1), dim=0)
                output, _ = predictor(data, similar_drug_embeddings, similar_target_embeddings)
            else:
                output, _ = predictor(data, drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
def train_predict():
    args.use_unimol = (args.unimol_switch == 1) or args.use_unimol
    args.use_cosine_annealing = (args.cosine_annealing_switch == 1) or args.use_cosine_annealing
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    print(f"Using ESM2 model: {args.esm_model} (output dimension: {ESM_DIMS[args.esm_model]})")
    if args.use_unimol:
        print(f"Using UniMol model: {args.unimol_model} (output dimension: {UNIMOL_DIMS[args.unimol_model]})")
        print(f"Available UniMol models: {list(UNIMOL_DIMS.keys())}")
        print(f"Selected drug_hidden_dim: {UNIMOL_DIMS[args.unimol_model]}")
    else:
        print("Using traditional graph-based features for drugs")
    print(f"Experimental scenario: {args.scenario}")
    print(f"Using heterogeneous graph neural network: {args.use_hetero}")
    print(f"Fallback to homogeneous model if needed: {args.fallback_to_homogeneous}")
    if args.scenario != 'S1':
        print(f"\nCold-start scenario settings:")
        print(f"  - Embedding enhancement: {'Enabled' if args.enable_embedding_enhancement else 'Disabled'}")
        print(f"  - Domain adaptation epochs: {args.cold_start_adapt_epochs}")
        print(f"  - Weight alpha for similarity: {args.weight_alpha}")
        print(f"  - Boost factor for top similar entities: {args.boost_factor}")
        if args.scenario in ['S2', 'S4']:
            print(f"  - Similar drugs to consider: {args.drug_sim_k}")
        if args.scenario in ['S3', 'S4']:
            print(f"  - Similar targets to consider: {args.target_sim_k}")
    if args.use_cosine_annealing:
        print(f"\nUsing Cosine Annealing learning rate scheduler:")
        print(f"  - Minimum LR: {args.lr_min}")
        print(f"  - Maximum LR: {args.lr_max}")
        print(f"  - Cycle length: {args.cycle_epochs} epochs")
        print(f"  - Warmup: {args.warmup_epochs} epochs")
    else:
        print(f"\nUsing fixed learning rate: {args.lr}")
    save_dir = f'checkpoints/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)

    affinity_mat = load_data(args.dataset)
    print(f"\nOriginal data statistics:")
    print(f"Total number of drugs: {affinity_mat.shape[0]}")
    print(f"Total number of targets: {affinity_mat.shape[1]}")
    print(f"Total number of drug-target pairs: {affinity_mat.shape[0] * affinity_mat.shape[1]}")
    print(f"Number of known interactions: {np.sum(~np.isnan(affinity_mat))}")
    
    if args.scenario == 'S1':
        train_data, test_data, affinity_graph, drug_pos, target_pos = process_data(
            affinity_mat, args.dataset, args.num_pos, args.pos_threshold, args.scenario, args.fold)
        drug_test_map = None
        drug_test_map_weight_norm = None
        target_test_map = None
        target_test_map_weight_norm = None
    elif args.scenario == 'S2':
        train_data, test_data, affinity_graph, drug_pos, target_pos, drug_test_map, drug_test_map_weight_norm = process_data(
            affinity_mat, args.dataset, args.num_pos, args.pos_threshold, args.scenario, args.fold, args.drug_sim_k)
        target_test_map = None
        target_test_map_weight_norm = None
        print(f"Using {args.drug_sim_k} similar drugs for prediction")
    elif args.scenario == 'S3':
        train_data, test_data, affinity_graph, drug_pos, target_pos, target_test_map, target_test_map_weight_norm = process_data(
            affinity_mat, args.dataset, args.num_pos, args.pos_threshold, args.scenario, args.fold, target_sim_k=args.target_sim_k)
        drug_test_map = None
        drug_test_map_weight_norm = None
        print(f"Using {args.target_sim_k} similar targets for prediction")
    elif args.scenario == 'S4':
        train_data, test_data, affinity_graph, drug_pos, target_pos, drug_test_map, drug_test_map_weight_norm, target_test_map, target_test_map_weight_norm = process_data(
            affinity_mat, args.dataset, args.num_pos, args.pos_threshold, args.scenario, args.fold, 
            args.drug_sim_k, args.target_sim_k)
        print(f"Using {args.drug_sim_k} similar drugs and {args.target_sim_k} similar targets for prediction")
    else:
        raise ValueError(f"Unsupported scenario: {args.scenario}")
    
    print(f"\nAfter {args.scenario} split:")
    print(f"Training set - Number of drug-target pairs: {len(train_data)}")
    print(f"Test set - Number of drug-target pairs: {len(test_data)}")
    
    train_drugs = len(set([data.drug_id.item() for data in train_data]))
    train_targets = len(set([data.target_id.item() for data in train_data]))
    print(f"Training set - Unique drugs: {train_drugs}, Unique targets: {train_targets}")
    
    test_drugs = len(set([data.drug_id.item() for data in test_data]))
    test_targets = len(set([data.target_id.item() for data in test_data]))
    print(f"Test set - Unique drugs: {test_drugs}, Unique targets: {test_targets}")
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    
    drug_data = json.load(open(f'data/{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict)
    
    if args.use_unimol:
        drug_features = get_drug_unimol_features(drug_data, args.unimol_model, args.unimol_batch_size)
        drug_tensors = []
        for d in drug_features:
            drug_tensors.append(drug_features[d])
        drug_graphs_DataLoader = torch.stack(drug_tensors, dim=0)

    
    print("\nModel preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    target_graphs_dict = get_target_molecule_graph(
        json.load(open(f'data/{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), 
        args.dataset,
        model_name=args.esm_model)
    
    target_tensors = []
    for t in target_graphs_dict:
        target_tensors.append(target_graphs_dict[t])
    target_graphs_DataLoader = torch.cat(target_tensors, dim=0).to(device)
    print(f"Target embeddings shape after loading: {target_graphs_DataLoader.shape}")

    drug_input_dim = UNIMOL_DIMS[args.unimol_model] if args.use_unimol else 78
    try:
        if args.use_hetero:
            print("Initializing heterogeneous graph neural network model...")
            model = UEC2DTA(tau=args.tau,
                        lam=args.lam,
                        ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                        d_ms_dims=[drug_input_dim, drug_input_dim, drug_input_dim * 2, 256] if args.use_unimol else [78, 78, 78 * 2, 256],
                        embedding_dim=128,
                        dropout_rate=args.edge_dropout_rate,
                        use_unimol=args.use_unimol,
                        drug_hidden_dim=drug_input_dim,
                        target_hidden_dim=ESM_DIMS[args.esm_model],
                        use_hetero=True)
            
            device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            print("Testing heterogeneous model with a forward pass...")
            with torch.no_grad():
                if isinstance(drug_graphs_DataLoader, torch.Tensor):
                    drug_graph_batchs = drug_graphs_DataLoader.to(device)
                else:
                    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
                
                target_graphs_DataLoader_device = target_graphs_DataLoader.to(device)
                drug_pos = drug_pos.to(device)
                target_pos = target_pos.to(device)
                affinity_graph = affinity_graph.to(device)

                _, _, _ = model(affinity_graph, drug_graph_batchs, target_graphs_DataLoader_device, drug_pos, target_pos)
                print("Heterogeneous model forward pass successful!")
        else:
            print("Using homogeneous graph neural network as specified...")
            model = UEC2DTA(tau=args.tau,
                        lam=args.lam,
                        ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                        d_ms_dims=[drug_input_dim, drug_input_dim, drug_input_dim * 2, 256] if args.use_unimol else [78, 78, 78 * 2, 256],
                        embedding_dim=128,
                        dropout_rate=args.edge_dropout_rate,
                        use_unimol=args.use_unimol,
                        drug_hidden_dim=drug_input_dim,
                        target_hidden_dim=ESM_DIMS[args.esm_model],
                        use_hetero=False)
    except Exception as e:
        print(f"Error initializing or testing heterogeneous model: {e}")
        if args.fallback_to_homogeneous:
            print("Falling back to homogeneous graph neural network...")
            model = UEC2DTA(tau=args.tau,
                       lam=args.lam,
                       ns_dims=[affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                       d_ms_dims=[drug_input_dim, drug_input_dim, drug_input_dim * 2, 256] if args.use_unimol else [78, 78, 78 * 2, 256],
                       embedding_dim=128,
                       dropout_rate=args.edge_dropout_rate,
                       use_unimol=args.use_unimol,
                       drug_hidden_dim=drug_input_dim,
                       target_hidden_dim=ESM_DIMS[args.esm_model],
                       use_hetero=False)
        else:
            raise e
    
    predictor = PredictModule()
    
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = model.to(device)
    predictor = predictor.to(device)
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    affinity_graph = affinity_graph.to(device)
    
    if drug_test_map is not None:
        drug_test_map = torch.tensor(drug_test_map, dtype=torch.long).to(device)
        drug_test_map_weight_norm = torch.tensor(drug_test_map_weight_norm, dtype=torch.float).to(device)
    
    if target_test_map is not None:
        target_test_map = torch.tensor(target_test_map, dtype=torch.long).to(device)
        target_test_map_weight_norm = torch.tensor(target_test_map_weight_norm, dtype=torch.float).to(device)
    
    if args.use_unimol:
        drug_graphs_DataLoader = drug_graphs_DataLoader.to(device)
    else:
        drug_graph_batchs = []
        for graph in drug_graphs_DataLoader:
            graph = graph.to(device)
            drug_graph_batchs.append(graph)
        drug_graphs_DataLoader = drug_graph_batchs
    if not isinstance(target_graphs_DataLoader, torch.Tensor):
        target_graphs_DataLoader = torch.tensor(target_graphs_DataLoader).to(device)
    else:
        target_graphs_DataLoader = target_graphs_DataLoader.to(device)

    if args.mode == 'test':
        if args.model_path is None:
            model_files = [f for f in os.listdir(save_dir) if f.startswith(f'{args.dataset}_{args.esm_model}_best')]
            if not model_files:
                raise FileNotFoundError(f"No model checkpoints found in {save_dir}")
            args.model_path = os.path.join(save_dir, model_files[0])
        model, predictor, _ = load_best_model(args.model_path, model, predictor, device)
        
        print("\nEvaluating model on test set...")
        try:
            G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,
                    affinity_graph, drug_pos, target_pos, args.scenario, drug_test_map, drug_test_map_weight_norm,
                    target_test_map, target_test_map_weight_norm, args.weight_alpha, args.boost_factor,
                    enhance_embeddings=args.enable_embedding_enhancement)
            mse, rm2, ci, pearson = model_evaluate(G, P)
            print("\nTest set performance:")
            print(f"MSE: {mse:.4f}")
            print(f"RM2: {rm2:.4f}")
            print(f"CI: {ci:.4f}")
            print(f"Pearson: {pearson:.4f}")
            print(f"Embedding enhancement: {'Enabled' if args.enable_embedding_enhancement else 'Disabled'}")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            if args.enable_embedding_enhancement:
                print("Retrying without embedding enhancement...")
                try:
                    G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader,
                          affinity_graph, drug_pos, target_pos, args.scenario, drug_test_map, drug_test_map_weight_norm,
                          target_test_map, target_test_map_weight_norm, args.weight_alpha, args.boost_factor,
                          enhance_embeddings=False)
                    mse, rm2, ci, pearson = model_evaluate(G, P)
                    print("\nTest set performance (without embedding enhancement):")
                    print(f"MSE: {mse:.4f}")
                    print(f"RM2: {rm2:.4f}")
                    print(f"CI: {ci:.4f}")
                    print(f"Pearson: {pearson:.4f}")
                except Exception as e:
                    print(f"Test still failed after disabling enhancement: {e}")
                    print("Try adjusting the model parameters or using a different scenario.")
            else:
                print("Try adjusting the model parameters or using a different scenario.")
        return

    print("Start training...")
    best_mse = float('inf')
    best_rm2 = float('-inf')
    best_combined_score = float('-inf')
    best_epoch = 0
    best_model_path = None
    start_time = time.time()
    
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), 
        lr=args.lr_max if args.use_cosine_annealing else args.lr, 
        weight_decay=0)
    
    scheduler = None
    if args.use_cosine_annealing:
        if args.warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < args.warmup_epochs:
                    return args.lr_min / args.lr_max + epoch * ((1.0 - args.lr_min / args.lr_max) / args.warmup_epochs)
                else:
                    progress = (epoch - args.warmup_epochs) / (args.cycle_epochs - 1)
                    return args.lr_min / args.lr_max + 0.5 * (1.0 - args.lr_min / args.lr_max) * (1 + np.cos(np.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.cycle_epochs, eta_min=args.lr_min)
    if isinstance(drug_graphs_DataLoader, torch.Tensor):
        drug_graph_batchs = drug_graphs_DataLoader.to(device)
    else:
        drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graphs_DataLoader = target_graphs_DataLoader.to(device)
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)
    affinity_graph = affinity_graph.to(device)
    if drug_test_map is not None:
        drug_test_map = drug_test_map.to(device)
        drug_test_map_weight_norm = drug_test_map_weight_norm.to(device)
    if target_test_map is not None:
        target_test_map = target_test_map.to(device)
        target_test_map_weight_norm = target_test_map_weight_norm.to(device)
    
    model = model.to(device)
    predictor = predictor.to(device)
    
    train_losses = []

    for epoch in range(args.epochs):
        avg_loss = train(model, predictor, device, train_loader, drug_graph_batchs, 
                         target_graphs_DataLoader, args.lr, epoch+1, args.batch_size, 
                         affinity_graph, drug_pos, target_pos, optimizer, scheduler)
        train_losses.append(avg_loss)
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            if args.scenario != 'S1' and args.cold_start_adapt_epochs > 0:
                print(f"\nPerforming domain adaptation for {args.scenario} cold-start scenario...")
                model.train()
                domain_optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.lr * 0.1
                )
                for adapt_epoch in range(args.cold_start_adapt_epochs):
                    model.train()
                    domain_optimizer.zero_grad()
                    try:
                        ssl_loss, _, _ = model(
                            affinity_graph.to(device), 
                            drug_graph_batchs,
                            target_graphs_DataLoader, 
                            drug_pos, 
                            target_pos,
                            args.scenario,  
                            enhance_embeddings=args.enable_embedding_enhancement  
                        )
                        
                        ssl_loss.backward()
                        domain_optimizer.step()
                        print(f"  Domain adaptation epoch {adapt_epoch+1}/{args.cold_start_adapt_epochs}, SSL loss: {ssl_loss.item():.4f}")
                    except Exception as e:
                        print(f"Error during domain adaptation: {e}")
                        if args.enable_embedding_enhancement:
                            try:
                                print("Retrying without embedding enhancement...")
                                domain_optimizer.zero_grad()
                                ssl_loss, _, _ = model(
                                    affinity_graph.to(device), 
                                    drug_graph_batchs,
                                    target_graphs_DataLoader, 
                                    drug_pos, 
                                    target_pos,
                                    args.scenario,
                                    enhance_embeddings=False 
                                )
                                ssl_loss.backward()
                                domain_optimizer.step()
                                print(f"  Domain adaptation epoch {adapt_epoch+1}/{args.cold_start_adapt_epochs}, SSL loss: {ssl_loss.item():.4f} (without enhancement)")
                            except Exception as e:
                                print(f"Still failed after disabling enhancement: {e}")
                                print("Skipping domain adaptation and proceeding with evaluation.")
                                break
                        else:
                            print("Skipping domain adaptation and proceeding with evaluation.")
                            break
                
                print("Domain adaptation completed.")
                model.eval()
            
            try:
                G, P = test(model, predictor, device, test_loader, drug_graph_batchs, target_graphs_DataLoader,
                          affinity_graph, drug_pos, target_pos, args.scenario, drug_test_map, drug_test_map_weight_norm,
                          target_test_map, target_test_map_weight_norm, args.weight_alpha, args.boost_factor,
                          enhance_embeddings=args.enable_embedding_enhancement)

                mse, rm2, ci, pearson = model_evaluate(G, P)
                print(f"epoch: {epoch+1}, MSE: {mse:.4f}, RM2: {rm2:.4f}, CI: {ci:.4f}, Pearson: {pearson:.4f}, Loss: {avg_loss:.4f}" + 
                      ("" if not args.enable_embedding_enhancement else " (with enhancement)"))
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if args.enable_embedding_enhancement:
                    try:
                        print("Retrying test without embedding enhancement...")
                        G, P = test(model, predictor, device, test_loader, drug_graph_batchs, target_graphs_DataLoader,
                              affinity_graph, drug_pos, target_pos, args.scenario, drug_test_map, drug_test_map_weight_norm,
                              target_test_map, target_test_map_weight_norm, args.weight_alpha, args.boost_factor,
                              enhance_embeddings=False)
                        
                        mse, rm2, ci, pearson = model_evaluate(G, P)
                        print(f"epoch: {epoch+1}, MSE: {mse:.4f}, RM2: {rm2:.4f}, CI: {ci:.4f}, Pearson: {pearson:.4f}, Loss: {avg_loss:.4f} (without enhancement)")
                    except Exception as e:
                        print(f"Test still failed after disabling enhancement: {e}")
                        print("Skipping this evaluation. Please check model configuration.")
                        continue
                else:
                    print("Skipping this evaluation. Please check model configuration.")
                    continue

            normalized_mse = 1.0 / (1.0 + mse)
            combined_score = normalized_mse + rm2
            
            if combined_score > best_combined_score:
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                best_mse = mse
                best_rm2 = rm2
                best_combined_score = combined_score
                best_epoch = epoch + 1

                current_time = time.time()
                training_seconds = current_time - start_time 
                
                model_name = (f'{args.dataset}_{args.esm_model}_'
                             f'{"unimol_" + args.unimol_model if args.use_unimol else "nounimol"}_'
                             f'best_mse{best_mse:.4f}_rm2{best_rm2:.4f}_'
                             f'CI{ci:.4f}_Pearson{pearson:.4f}_'
                             f'lam{args.lam}_tau{args.tau}_pos{args.num_pos}_'
                             f'epoch{best_epoch}.pt')
                best_model_path = os.path.join(save_dir, model_name)
                
                try:
                    _, drug_embeddings, target_embeddings = model(
                        affinity_graph.to(device), 
                        drug_graph_batchs, 
                        target_graphs_DataLoader, 
                        drug_pos, 
                        target_pos,
                        args.scenario,
                        enhance_embeddings=args.enable_embedding_enhancement
                    )
                    
                    test_drug_indices = [data.drug_id.item() for data in test_loader.dataset]
                    test_target_indices = [data.target_id.item() for data in test_loader.dataset]
                except Exception as e:
                    print(f"Warning: Error getting embeddings for checkpoint: {e}")
                    print("Using dummy embeddings for checkpoint")
                    drug_embeddings = torch.zeros(1)
                    target_embeddings = torch.zeros(1)
                    test_drug_indices = []
                    test_target_indices = []
                
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'predictor_state_dict': predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'mse': best_mse,
                    'rm2': best_rm2,
                    'ci': ci,
                    'pearson': pearson,
                    'combined_score': combined_score,
                    'training_seconds': training_seconds,  
                    'train_losses': train_losses,  
                    
                    'args': vars(args),  
                    'model_config': {
                        'tau': args.tau,
                        'lam': args.lam,
                        'ns_dims': [affinity_graph.num_drug + affinity_graph.num_target + 2, 512, 256],
                        'd_ms_dims': [drug_input_dim, drug_input_dim, drug_input_dim * 2, 256] if args.use_unimol else [78, 78, 78 * 2, 256],
                        'embedding_dim': 128,
                        'dropout_rate': args.edge_dropout_rate,
                        'use_unimol': args.use_unimol,
                        'drug_hidden_dim': drug_input_dim,
                        'target_hidden_dim': ESM_DIMS[args.esm_model],
                        'use_hetero': args.use_hetero,
                        'use_cosine_annealing': args.use_cosine_annealing
                    },
                    
                    'affinity_data': {
                        'drug_target_pairs': list(zip(test_drug_indices, test_target_indices)),  
                        'predicted_affinities': P,  
                        'true_affinities': G,  
                        'drug_ids': test_drug_indices,  
                        'target_ids': test_target_indices,  
                    },
                    
                    'num_drugs': affinity_graph.num_drug,
                    'num_targets': affinity_graph.num_target,
                    'esm_model_name': args.esm_model,
                    'scenario': args.scenario,
                    'drug_sim_k': args.drug_sim_k if args.scenario in ['S2', 'S4'] else None,
                    'target_sim_k': args.target_sim_k if args.scenario in ['S3', 'S4'] else None,
                }, best_model_path)
                print(f"Saved new best model with MSE: {best_mse:.4f}, RM2: {best_rm2:.4f}, CI: {ci:.4f}, Pearson: {pearson:.4f}")
                print(f"Training time so far: {training_seconds:.2f} seconds")

    total_training_seconds = time.time() - start_time
    print('\nTraining completed')
    print(f"Total training time: {total_training_seconds:.2f} seconds")
    print(f"Best performance at epoch {best_epoch}:")
    print(f"MSE: {best_mse:.4f}")
    print(f"RM2: {best_rm2:.4f}")
    print(f"Best model saved as: {os.path.basename(best_model_path)}")
    print(f"Model hyperparameters:")
    print(f"  - Lambda: {args.lam}")
    print(f"  - Tau: {args.tau}")
    print(f"  - Num positive pairs: {args.num_pos}")
    print(f"  - UniMol: {'Enabled (' + args.unimol_model + ')' if args.use_unimol else 'Disabled'}")
    
    if args.use_cosine_annealing and scheduler:
        import matplotlib.pyplot as plt
        lr_history = []
        scheduler_copy = type(scheduler)(optimizer, **scheduler.state_dict())
        for _ in range(args.epochs):
            lr_history.append(scheduler_copy.get_last_lr()[0])
            scheduler_copy.step()
        
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, args.epochs + 1), lr_history)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        lr_plot_path = os.path.join(save_dir, f'{args.dataset}_lr_schedule.png')
        plt.savefig(lr_plot_path)
        print(f"Learning rate schedule plot saved to {lr_plot_path}")

        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(train_losses) + 1), train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        loss_plot_path = os.path.join(save_dir, f'{args.dataset}_training_loss.png')
        plt.savefig(loss_plot_path)
        print(f"Training loss plot saved to {loss_plot_path}")

if __name__ == '__main__':
    train_predict()
