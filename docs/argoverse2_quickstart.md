# Argoverse2 クイックスタート

## 1. データセット取得

[Argoverse2 公式サイト](https://www.argoverse.org/av2.html) から Sensor Dataset (val) をダウンロード。

## 2. ディレクトリ配置

```
datasets/Argoverse2/
└── sensor/
    └── val/
        ├── <scene_id>/
        │   ├── sensors/lidar/...
        │   └── ...
        └── ...
```

## 3. 前処理

```bash
conda activate sf_tv
python dataprocess/extract_av2.py \
  --av2_type sensor \
  --data_mode val \
  --argo_dir /path/to/Argoverse2 \
  --output_dir /path/to/Argoverse2/preprocess_v2 \
  --nproc 8
```

出力: `preprocess_v2/sensor/val/` に HDF5 ファイルが生成される。

## 4. 推論

```bash
python save.py \
  model=sf_voxel_model \
  checkpoint=checkpoints/voteflow_best_m8n128_ori.ckpt \
  dataset_path=/path/to/Argoverse2/preprocess_v2/sensor/val \
  res_name=flow_est \
  gpus=[0] \
  batch_size=4
```

## 5. 可視化

```bash
python tools/visualization.py vis \
  --data_dir /path/to/Argoverse2/preprocess_v2/sensor/val \
  --res_name flow_est \
  --start_id 0 \
  --point_size 2.0
```

操作:
- `SPACE`: 再生/停止
- `N`: 次フレーム
- `ESC`/`Q`: 終了

