"""
# Created: 2023-12-26 12:41
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: produce flow based on model predict and write into the dataset, 
#              then use `tools/visualization.py` to visualize the flow.
"""

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra, wandb, os, sys, time
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from src.dataset import HDF5Dataset, collate_fn_pad_test
from src.trainer import ModelWrapper
from src.utils import bc

@hydra.main(version_base=None, config_path="conf", config_name="save")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)

    if cfg.res_name is None:
        cfg.res_name = cfg.checkpoint.split("/")[-1].split(".")[0]
        print(f"{bc.BOLD}NOTE{bc.ENDC}: res_name is not specified, use {bc.OKBLUE}{cfg.res_name}{bc.ENDC} as default.")

    checkpoint_params = DictConfig(torch.load(cfg.checkpoint)["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output
    cfg.model.update(checkpoint_params.cfg.model)
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)

    # wrap forward to measure per-call latency
    if hasattr(mymodel, "model") and hasattr(mymodel.model, "forward"):
        orig_forward = mymodel.model.forward
        def timed_forward(*args, **kwargs):
            t0 = time.time()
            out = orig_forward(*args, **kwargs)
            dt = (time.time() - t0) * 1000
            # print(f"[ForwardTime] {dt:.2f} ms")
            return out
        mymodel.model.forward = timed_forward

    wandb_logger = WandbLogger(save_dir=output_dir,
                               project=f"sceneflow_translation_voting", 
                               name=f"{cfg.output}",
                               offline=True)
    
    # set up dataset/dataloader explicitly to measure runtime
    pair_stride = cfg.pair_stride if 'pair_stride' in cfg else 1
    dataset = HDF5Dataset(
        cfg.dataset_path,
        n_frames=checkpoint_params.cfg.num_frames if 'num_frames' in checkpoint_params.cfg else 2,
        pair_stride=pair_stride
    )
    batch_size = cfg.batch_size if 'batch_size' in cfg else 1
    base_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn_pad_test, num_workers=4, pin_memory=True)

    # wrap dataloader with tqdm to log per-batch progress
    class TqdmDataLoader:
        def __init__(self, loader):
            self.loader = loader
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            return iter(tqdm(self.loader, total=len(self.loader), desc="test", ncols=100))

        @property
        def dataset(self):  # preserve dataset attribute if accessed by Lightning
            return self.loader.dataset

    dataloader = TqdmDataLoader(base_loader)
    
    # Lightningのprogress barはオフにし、こちらのtqdm（dataloader側）だけ表示
    trainer = pl.Trainer(logger=wandb_logger, devices=cfg.gpus, enable_progress_bar=False)
    # NOTE(Qingwen): search & check in pl_model.py : def test_step(self, batch, res_dict)
    t0 = time.time()
    trainer.test(model = mymodel, dataloaders = dataloader)
    elapsed = time.time() - t0
    num_samples = len(dataset)
    print(f"[Timing] test finished: {elapsed:.2f}s for {num_samples} samples ({num_samples/elapsed if elapsed>0 else 0:.2f} samples/s)")
    wandb.finish()

if __name__ == "__main__":
    main()