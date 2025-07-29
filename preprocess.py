import pickle
import json
import torch
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
import esm
import unimol_tools

from torch_geometric import data as DATA
from collections import OrderedDict
from rdkit import Chem
from metrics import DTADataset, sparse_mx_to_torch_sparse_tensor, minMaxNormalize, denseAffinityRefine



def load_data(dataset):
    affinity = pickle.load(open('data/' + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)

    return affinity

def process_data(affinity_mat, dataset, num_pos, pos_threshold, scenario='S1', fold=-100, drug_sim_k=2, target_sim_k=7):
    dataset_path = 'data/' + dataset + '/'

    if scenario == 'S1':
        train_file = json.load(open(dataset_path + 'S1_train_set.txt'))
        train_index = []
        for i in range(len(train_file)):
            train_index += train_file[i]
        test_index = json.load(open(dataset_path + 'S1_test_set.txt'))
    elif scenario == 'S2':
        drug_train_fold_origin = json.load(open(dataset_path + 'S2_train_set.txt'))
        drug_train_folds = []
        for i in range(len(drug_train_fold_origin)):
            if i != fold:
                drug_train_folds += drug_train_fold_origin[i]
        drug_test_fold = json.load(open(dataset_path + 'S2_test_set.txt')) if fold == -100 else drug_train_fold_origin[fold]
        drug_mask_fold = json.load(open(dataset_path + 'S2_mask_set.txt')) if fold == -100 else json.load(open(dataset_path + 'S2_test_set.txt')) + json.load(open(dataset_path + 'S2_mask_set.txt'))
        train_affinity = affinity_mat.copy()
        train_affinity[drug_mask_fold, :] = np.nan  
        train_affinity = train_affinity[drug_train_folds, :]
        
        test_affinity = affinity_mat[drug_test_fold, :]

        train_rows, train_cols = np.where(np.isnan(train_affinity) == False)
        train_Y = train_affinity[train_rows, train_cols]
        train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

        test_rows, test_cols = np.where(np.isnan(test_affinity) == False)
        test_Y = test_affinity[test_rows, test_cols]
        test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

        train_affinity[np.isnan(train_affinity) == True] = 0
        affinity_graph, drug_pos, target_pos = get_affinity_graph(dataset, train_affinity, num_pos, pos_threshold, scenario)
        

        drug_sim = np.loadtxt(f"data/{dataset}/drug-drug-sim.txt", delimiter=",")
        drug_test_train_sim = drug_sim[drug_test_fold, :]
        drug_test_train_sim[:, drug_test_fold + drug_mask_fold] = -1  

        drug_count = affinity_mat.shape[0]
        drug_train_count = len(drug_train_folds)
        drug_test_train_map = np.argpartition(drug_test_train_sim, -drug_sim_k, 1)[:, -drug_sim_k:]
        drug_train_map = np.full(drug_count, -1)
        drug_train_map[drug_train_folds] = np.arange(drug_train_count)
        drug_test_map = drug_train_map[drug_test_train_map]

        drug_test_map_weight = drug_test_train_sim[
            np.tile(np.expand_dims(np.arange(drug_test_train_sim.shape[0]), 0), (drug_sim_k, 1)).transpose(), 
            drug_test_train_map
        ]
        drug_test_map_weight_sum = np.expand_dims(np.sum(drug_test_map_weight, axis=1), axis=1)
        drug_test_map_weight_norm = np.expand_dims(drug_test_map_weight / drug_test_map_weight_sum, axis=2)
        
        return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos, drug_test_map, drug_test_map_weight_norm
        
    elif scenario == 'S3':
        target_train_fold_origin = json.load(open(dataset_path + 'S3_train_set.txt'))
        target_train_folds = []
        for i in range(len(target_train_fold_origin)):
            if i != fold:
                target_train_folds += target_train_fold_origin[i]
        target_test_fold = json.load(open(dataset_path + 'S3_test_set.txt')) if fold == -100 else target_train_fold_origin[fold]
        target_mask_fold = json.load(open(dataset_path + 'S3_mask_set.txt')) if fold == -100 else json.load(open(dataset_path + 'S3_test_set.txt')) + json.load(open(dataset_path + 'S3_mask_set.txt'))

        train_affinity = affinity_mat.copy()
        train_affinity[:, target_mask_fold] = np.nan  
        train_affinity = train_affinity[:, target_train_folds]
     
        test_affinity = affinity_mat[:, target_test_fold]

        train_rows, train_cols = np.where(np.isnan(train_affinity) == False)
        train_Y = train_affinity[train_rows, train_cols]
        train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

        test_rows, test_cols = np.where(np.isnan(test_affinity) == False)
        test_Y = test_affinity[test_rows, test_cols]
        test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

        train_affinity[np.isnan(train_affinity) == True] = 0
        affinity_graph, drug_pos, target_pos = get_affinity_graph(dataset, train_affinity, num_pos, pos_threshold, scenario)
        
        target_sim = np.loadtxt(f"data/{dataset}/target-target-sim.txt", delimiter=",")
        target_test_train_sim = target_sim[target_test_fold, :]
        target_test_train_sim[:, target_test_fold + target_mask_fold] = -1  

        target_count = affinity_mat.shape[1]
        target_train_count = len(target_train_folds)

        target_test_train_map = np.argpartition(target_test_train_sim, -target_sim_k, 1)[:, -target_sim_k:]
        target_train_map = np.full(target_count, -1)
        target_train_map[target_train_folds] = np.arange(target_train_count)
        target_test_map = target_train_map[target_test_train_map]

        target_test_map_weight = target_test_train_sim[
            np.tile(np.expand_dims(np.arange(target_test_train_sim.shape[0]), 0), (target_sim_k, 1)).transpose(), 
            target_test_train_map
        ]
        target_test_map_weight_sum = np.expand_dims(np.sum(target_test_map_weight, axis=1), axis=1)
        target_test_map_weight_norm = np.expand_dims(target_test_map_weight / target_test_map_weight_sum, axis=2)
        
        return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos, target_test_map, target_test_map_weight_norm
        
    elif scenario == 'S4':

        drug_train_fold_origin = json.load(open(dataset_path + 'S2_train_set.txt'))
        drug_train_folds = []
        for i in range(len(drug_train_fold_origin)):
            drug_train_folds += drug_train_fold_origin[i]
        drug_test_fold = json.load(open(dataset_path + 'S2_test_set.txt'))
        drug_mask_fold = json.load(open(dataset_path + 'S2_mask_set.txt'))
        
        target_train_fold_origin = json.load(open(dataset_path + 'S3_train_set.txt'))
        target_train_folds = []
        for i in range(len(target_train_fold_origin)):
            target_train_folds += target_train_fold_origin[i]
        target_test_fold = json.load(open(dataset_path + 'S3_test_set.txt'))
        target_mask_fold = json.load(open(dataset_path + 'S3_mask_set.txt'))

        train_affinity = affinity_mat.copy()
        train_affinity[drug_mask_fold, :] = np.nan  
        train_affinity[:, target_mask_fold] = np.nan  
        train_affinity = train_affinity[drug_train_folds, :][:, target_train_folds]
        
        test_affinity = affinity_mat[drug_test_fold, :][:, target_test_fold]

        train_rows, train_cols = np.where(np.isnan(train_affinity) == False)
        train_Y = train_affinity[train_rows, train_cols]
        train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

        test_rows, test_cols = np.where(np.isnan(test_affinity) == False)
        test_Y = test_affinity[test_rows, test_cols]
        test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

        train_affinity[np.isnan(train_affinity) == True] = 0
        affinity_graph, drug_pos, target_pos = get_affinity_graph(dataset, train_affinity, num_pos, pos_threshold, scenario)
        
        drug_sim = np.loadtxt(f"data/{dataset}/drug-drug-sim.txt", delimiter=",")
        drug_test_train_sim = drug_sim[drug_test_fold, :]
        drug_test_train_sim[:, drug_test_fold + drug_mask_fold] = -1

        drug_count = affinity_mat.shape[0]
        drug_train_count = len(drug_train_folds)
        drug_test_train_map = np.argpartition(drug_test_train_sim, -drug_sim_k, 1)[:, -drug_sim_k:]
        drug_train_map = np.full(drug_count, -1)
        drug_train_map[drug_train_folds] = np.arange(drug_train_count)
        drug_test_map = drug_train_map[drug_test_train_map]

        drug_test_map_weight = drug_test_train_sim[
            np.tile(np.expand_dims(np.arange(drug_test_train_sim.shape[0]), 0), (drug_sim_k, 1)).transpose(), 
            drug_test_train_map
        ]
        drug_test_map_weight_sum = np.expand_dims(np.sum(drug_test_map_weight, axis=1), axis=1)
        drug_test_map_weight_norm = np.expand_dims(drug_test_map_weight / drug_test_map_weight_sum, axis=2)
        
        target_sim = np.loadtxt(f"data/{dataset}/target-target-sim.txt", delimiter=",")
        target_test_train_sim = target_sim[target_test_fold, :]
        target_test_train_sim[:, target_test_fold + target_mask_fold] = -1

        target_count = affinity_mat.shape[1]
        target_train_count = len(target_train_folds)
        target_test_train_map = np.argpartition(target_test_train_sim, -target_sim_k, 1)[:, -target_sim_k:]
        target_train_map = np.full(target_count, -1)
        target_train_map[target_train_folds] = np.arange(target_train_count)
        target_test_map = target_train_map[target_test_train_map]

        target_test_map_weight = target_test_train_sim[
            np.tile(np.expand_dims(np.arange(target_test_train_sim.shape[0]), 0), (target_sim_k, 1)).transpose(), 
            target_test_train_map
        ]
        target_test_map_weight_sum = np.expand_dims(np.sum(target_test_map_weight, axis=1), axis=1)
        target_test_map_weight_norm = np.expand_dims(target_test_map_weight / target_test_map_weight_sum, axis=2)
        
        return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos, drug_test_map, drug_test_map_weight_norm, target_test_map, target_test_map_weight_norm
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    rows, cols = np.where(np.isnan(affinity_mat) == False)
    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity_mat = np.zeros_like(affinity_mat)
    train_affinity_mat[train_rows, train_cols] = train_Y
    affinity_graph, drug_pos, target_pos = get_affinity_graph(dataset, train_affinity_mat, num_pos, pos_threshold)

    return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos

def get_affinity_graph(dataset, adj, num_pos, pos_threshold, scenario='S1'):
    dataset_path = 'data/' + dataset + '/'
    num_drug, num_target = adj.shape[0], adj.shape[1]

    dt_ = adj.copy()
    dt_ = np.where(dt_ >= pos_threshold, 1.0, 0.0)
    dtd = np.matmul(dt_, dt_.T)
    dtd = dtd / dtd.sum(axis=-1).reshape(-1, 1)
    dtd = np.nan_to_num(dtd)
    dtd += np.eye(num_drug, num_drug)
    dtd = dtd.astype("float32")
    
    d_d_full = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')
    
    valid_drug_indices = np.arange(num_drug)
    if np.sum(adj) == 0: 
        valid_drug_indices = np.where(np.sum(adj, axis=1) != 0)[0]

    d_d = d_d_full[valid_drug_indices][:, valid_drug_indices]
    
    dAll = dtd + d_d

    drug_pos = np.zeros((len(valid_drug_indices), len(valid_drug_indices)))
    for i in range(len(dAll)):
        one = dAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-dAll[i, one])
            sele = one[oo[:num_pos]]
            drug_pos[i, sele] = 1
        else:
            drug_pos[i, one] = 1
    drug_pos = sp.coo_matrix(drug_pos)
    drug_pos = sparse_mx_to_torch_sparse_tensor(drug_pos)

    td_ = adj.T.copy()
    td_ = np.where(td_ >= pos_threshold, 1.0, 0.0)
    tdt = np.matmul(td_, td_.T)
    tdt = tdt / tdt.sum(axis=-1).reshape(-1, 1)
    tdt = np.nan_to_num(tdt)
    tdt += np.eye(num_target, num_target)
    tdt = tdt.astype("float32")

    t_t_full = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')
    
    valid_target_indices = np.arange(num_target)
    if np.sum(adj) == 0:  
        valid_target_indices = np.where(np.sum(adj.T, axis=1) != 0)[0]
    
    t_t = t_t_full[valid_target_indices][:, valid_target_indices]
    
    tAll = tdt + t_t
    target_pos = np.zeros((len(valid_target_indices), len(valid_target_indices)))
    for i in range(len(tAll)):
        one = tAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-tAll[i, one])
            sele = one[oo[:num_pos]]
            target_pos[i, sele] = 1
        else:
            target_pos[i, one] = 1
    target_pos = sp.coo_matrix(target_pos)
    target_pos = sparse_mx_to_torch_sparse_tensor(target_pos)

    if dataset == "davis":
        adj[adj != 0] -= 5
        adj_norm = minMaxNormalize(adj, 0)
    elif dataset == "kiba":

        if scenario in ['S2', 'S4']:
            print(f"Using optimized affinity refinement for {scenario} scenario")
            adj_refine = denseAffinityRefine(adj.T, 90)  # 降低target参数以减少噪声
            adj_refine = denseAffinityRefine(adj_refine.T, 40)
        else:  
            print(f"Using standard affinity refinement for {scenario} scenario")
            adj_refine = denseAffinityRefine(adj.T, 150)
            adj_refine = denseAffinityRefine(adj_refine.T, 40)
        adj_norm = minMaxNormalize(adj_refine, 0)
    adj_1 = adj_norm
    adj_2 = adj_norm.T
    adj = np.concatenate((
        np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
    ), 0)
    train_row_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_row_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    edge_weights = adj[train_row_ids, train_col_ids]
    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drug, 1)),
        np.tile(np.array([0, 1]), (num_target, 1))
    ), 0)
    adj_features = np.zeros_like(adj)
    adj_features[adj != 0] = 1
    features = np.concatenate((node_type_features, adj_features), 1)
    affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj),
                               edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_drug", num_drug)
    affinity_graph.__setitem__("num_target", num_target)

    return affinity_graph, drug_pos, target_pos

MODEL_LAYERS = {
    'esm2_t6_8M_UR50D': 6,
    'esm2_t12_35M_UR50D': 12,
    'esm2_t30_150M_UR50D': 30,
    'esm2_t33_650M_UR50D': 33,
    'esm2_t36_3B_UR50D': 36
}

UNIMOL_DIMS = {
    'unimolv1': 512,
    'unimolv2_84m': 768,
    'unimolv2_164m': 1024,
    'unimolv2_310m': 1280,
    'unimolv2_570m': 1536,
    'unimolv2_1.1B': 2048,
}

def get_drug_unimol_features(ligands, model_name="unimolv1", batch_size=32):
    
    print("\nInitializing UniMol model...")
   
    if model_name == "unimolv1":
        unimol_model_name = "unimolv1"
        unimol_model_size = None
    else:
        parts = model_name.split("_")
        unimol_model_name = parts[0]
        unimol_model_size = parts[1]
    
    from unimol_tools import UniMolRepr

    model = UniMolRepr(
        data_type='molecule',
        batch_size=batch_size,
        remove_hs=False,
        model_name=unimol_model_name,
        model_size=unimol_model_size if unimol_model_size else '84m'
    )
    print(f"UniMol model initialized: {model_name}")
    
    smiles_list = list(ligands.values())
    drug_ids = list(ligands.keys())

    print(f"Selected UniMol model: {model_name}")
    print(f"Expected output dimension: {UNIMOL_DIMS[model_name if model_name in UNIMOL_DIMS else 'unimolv1']}")
    
    drug_features = OrderedDict()
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        batch_ids = drug_ids[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(smiles_list) + batch_size - 1) // batch_size}")
        
        try:
            unimol_repr = model.get_repr(batch_smiles)
            
            for j, drug_id in enumerate(batch_ids):
                drug_features[drug_id] = torch.tensor(unimol_repr['cls_repr'][j])
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            for j, drug_id in enumerate(batch_ids):
                if drug_id not in drug_features:
                    dim = UNIMOL_DIMS[model_name if model_name in UNIMOL_DIMS else "unimolv1"]
                    drug_features[drug_id] = torch.zeros(dim)
    
    missing_drugs = set(drug_ids) - set(drug_features.keys())
    if missing_drugs:
        print(f"Warning: {len(missing_drugs)} drugs have no representation and will use zero vectors")
        dim = UNIMOL_DIMS[model_name if model_name in UNIMOL_DIMS else "unimolv1"]
        for drug_id in missing_drugs:
            drug_features[drug_id] = torch.zeros(dim)
    
    print(f"\nFinal drug embeddings count: {len(drug_features)}")
    sample_key = next(iter(drug_features))
    print(f"Drug embedding dimension: {drug_features[sample_key].shape}")
    
    return drug_features

def get_esm_model(model_name):
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    elif model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif model_name == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    batch_converter = alphabet.get_batch_converter()
    model.eval()  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, batch_converter, device


def get_target_molecule_graph(proteins, dataset, model_name="esm2_t33_650M_UR50D"):
    print("\nInitializing ESM2 model...")

    model, batch_converter, device = get_esm_model(model_name)
    print(f"ESM2 model initialized: {model_name}")
    
    last_layer = MODEL_LAYERS[model_name]
    print(f"Using representation from layer {last_layer}")

    target_graph = OrderedDict()
    for t in proteins.keys():
        data = [ (t, proteins[t]) ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.to(device=device, non_blocking=True)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[last_layer], return_contacts=False)
        token_embeddings = results["representations"][last_layer]

        sequence_embeddings = token_embeddings.mean(dim=1)
        print(f"Target {t} embedding shape:", sequence_embeddings.shape)
        
        target_graph[t] = sequence_embeddings

    target_tensors = []
    for t in target_graph:
        target_tensors.append(target_graph[t])

    target_embeddings = torch.cat(target_tensors, dim=0)
    print(f"\nFinal target embeddings shape: {target_embeddings.shape}")
    
    return target_graph


