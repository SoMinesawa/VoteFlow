# SemanticKITTI クイックスタート

## 1. データセット準備

SemanticKITTI データセットとPatchwork++地面マスクを以下の構造で配置：

```
data/dataset/semantickitti/dataset/
└── sequences/
    ├── 00/
    │   ├── velodyne/*.bin
    │   ├── poses.txt
    │   └── ...
    └── ...

data/users/minesawa/semantickitti/patchwork-plusplus/
└── sequences/
    ├── 00/
    │   └── predictions/*.label
    └── ...
```

## 2. 前処理

```bash
conda activate sf_tv

python dataprocess/extract_semantickitti.py \
  --kitti_dir data/dataset/semantickitti/dataset \
  --patchwork_dir data/users/minesawa/semantickitti/patchwork-plusplus \
  --output_dir data/dataset/semantickitti/voteflow_preprocess \
  --nproc 16
```

出力: `voteflow_preprocess/` に HDF5 ファイルが生成される。

## 3. 推論

```bash
HDF5_USE_FILE_LOCKING=FALSE python save.py --config-name=save_semantickitti \
  checkpoint=checkpoints/voteflow_best_m8n128_ori.ckpt \
  res_name=flow_est \
  batch_size=8
```

出力:
- `outputs/flow_est/{seq_id}/{frame_id}.pkl`
- `voteflow_preprocess/*.h5` に `flow_est` が追加

## 4. 可視化

```bash
python tools/visualization.py vis \
  --data_dir data/dataset/semantickitti/voteflow_preprocess \
  --res_name flow_est \
  --start_id 0 \
  --point_size 2.0
```

操作:
- `SPACE`: 再生/停止
- `N`: 次フレーム
- `ESC`/`Q`: 終了

## データ構造

### 前処理後
```
voteflow_preprocess/
├── 00.h5
│   └── 000000/
│       ├── lidar
│       ├── ground_mask  (Patchwork++)
│       └── pose
└── index_total.pkl
```

### 推論後
```
voteflow_preprocess/00.h5
└── 000000/
    ├── lidar
    ├── ground_mask
    ├── pose
    └── flow_est  ← 追加

outputs/flow_est/00/000000.pkl  ← バックアップ
```

