from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MSTCN2_Config:
    # General configs
    device: str = 'cuda:8'
    # Data configs
    dataset: str = 'gtea'
    split: str = '1'
    feature_dir: str = 'features'
    gt_dir: str = 'groundTruth'
    splits_dir: str = 'splits'
    # Model hyperparameters
    features_dim: int = 2048
    num_layers_PG: int = 11
    num_layers_R: int = 10
    num_R: int = 3
    # Training configs
    num_epochs: int = 100
    batch_size: int = 16
    lr: float = 0.0005
    num_f_maps: int = 64
    # Eval configs
    test_epoch: int = 100
    checkpoint_path_explicit: Optional[str] = None # explicitly specify checkpoint path, optional

configs: Dict[str, MSTCN2_Config] = {
    'default': MSTCN2_Config(),
}
