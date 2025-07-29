# UEC2DTA: A Contrastive Graph Framwork Integrating Pre-trained Uni-Mol and ESM2 for Drug-Target Affinity Prediction

UEC2DTA is a deep learning model for Drug-Target Affinity (DTA) prediction, designed to effectively handle various scenarios including cold-start problems where either drugs, targets, or both are previously unseen.

## Overview

UEC2DTA leverages heterogeneous graph neural networks, contrastive learning, and state-of-the-art protein and molecule representation models (ESM and UniMol) to predict binding affinity between drugs and protein targets. The model can handle four experimental scenarios:

- **S1**: Random entries (standard prediction)
- **S2**: Unseen drugs (predicting with novel compounds)
- **S3**: Unseen targets (predicting with novel proteins)
- **S4**: All unseen (both drugs and targets are novel)

## Key Features

- Heterogeneous graph neural network architecture for better representation learning
- Contrastive learning to enhance embedding quality
- Integration with ESM2 for protein sequence representation
- Integration with UniMol for drug molecule representation
- Cold-start domain adaptation for novel compounds and proteins
- Customizable model configuration for different datasets and requirements

## Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- ESM (Facebook AI's Evolutionary Scale Modeling)
- UniMol
- RDKit
- NumPy
- SciPy
- NetworkX

## Datasets

The model supports the following benchmark datasets:
- Davis
- KIBA

Data should be placed in the `data/` directory with the following structure:
```
data/
├── davis/
│   ├── affinities
│   ├── drug-drug-sim.txt
│   ├── target-target-sim.txt
│   ├── S1_train_set.txt
│   ├── S1_test_set.txt
│   └── ...
├── kiba/
│   ├── affinities
│   └── ...
```

## Usage

### Data Preprocessing

```python
python preprocess.py --dataset davis --esm_model esm2_t33_650M_UR50D
```

### Training

```python
python predict.py --dataset davis --scenario S1 --epochs 1000 --batch_size 128 --lr 0.0002 --edge_dropout_rate 0.2 --tau 0.8 --lam 0.5 --num_pos 5 --pos_threshold 8.0 --mode train
```

### Testing

```python
python predict.py --dataset davis --scenario S1 --mode test --model_path /path/to/checkpoint.pt
```

## Command Line Arguments

- `--cuda`: GPU device ID (default: 0)
- `--scenario`: Experimental scenario (S1, S2, S3, S4) (default: S1)
- `--dataset`: Dataset to use (davis, kiba) (default: davis)
- `--epochs`: Number of training epochs (default: 10000)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.0002)
- `--edge_dropout_rate`: Edge dropout rate for graph neural networks (default: 0.2)
- `--tau`: Temperature parameter for contrastive learning (default: 0.8)
- `--lam`: Balance parameter for contrastive learning (default: 0.5)
- `--num_pos`: Number of positive samples for contrastive learning (default: 5)
- `--pos_threshold`: Affinity threshold for positive samples (default: 8.0)
- `--esm_model`: ESM model version for protein representation (default: esm2_t33_650M_UR50D)
- `--unimol_model`: UniMol model version for drug representation (default: unimolv1)
- `--mode`: Training or testing mode (default: train)

## Model Architecture

UEC2DTA integrates several components:

1. **Drug Representation**: 
   - Graph-based representation using molecular structure
   - UniMol embedding for improved feature extraction

2. **Protein Representation**:
   - ESM2 embeddings for advanced protein sequence features

3. **Heterogeneous Graph Neural Network**:
   - Models interactions between drugs and targets
   - Handles drug-drug, target-target, and drug-target edges

4. **Contrastive Learning Module**:
   - Enhances embedding quality through positive/negative sample discrimination

5. **Cold-start Adaptation**:
   - Special handling for novel drugs/proteins through similarity-based mapping

## Evaluation Metrics

The model is evaluated using:
- Mean Squared Error (MSE)
- Concordance Index (CI)
- R² coefficient (RM2)
- Pearson correlation coefficient

## References

If you use this code or model, please cite:
```
[Citation information to be added]
``` 