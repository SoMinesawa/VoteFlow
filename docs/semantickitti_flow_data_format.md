# SemanticKITTI Scene Flow データフォーマット

VoteFlowで推定したScene Flowデータの構造と使用方法。

## ファイル構造

```
voteflow_preprocess_fixed/
├── 01.h5              # シーケンスデータ (HDF5)
└── index_total.pkl    # インデックス
```

## HDF5構造

```
01.h5
├── 000000/           # タイムスタンプ (フレーム番号)
│   ├── lidar         # (N, 3) float32 - 点群 [x, y, z]
│   ├── pose          # (4, 4) float32 - Velodyne座標系でのワールド姿勢
│   ├── ground_mask   # (N,) bool - 地面点フラグ
│   └── flow_est_fixed # (N, 3) float32 - 推定Scene Flow [dx, dy, dz]
├── 000001/
│   └── ...
```

## データ詳細

| キー | 形状 | 型 | 説明 |
|------|------|-----|------|
| `lidar` | (N, 3) | float32 | Velodyne座標系の点群 |
| `pose` | (4, 4) | float32 | フレームのワールド姿勢行列 |
| `ground_mask` | (N,) | bool | True = 地面点 |
| `flow_est_fixed` | (N, 3) | float32 | 次フレームへの3D移動ベクトル |

- **座標系**: すべてVelodyne（LiDAR）座標系
- **flow**: `pc_t + flow_est_fixed ≈ pc_t+1` の関係（エゴモーション込み）

> **注意**: HDF5内の`pose`はVelodyne座標系に変換済み。SemanticKITTIの元の`poses.txt`はCamera0座標系のため、直接使用不可。ワールド座標変換には必ずHDF5の`pose`を使用すること。

## 読み込み例

```python
import h5py

with h5py.File('01.h5', 'r') as f:
    timestamp = '000100'
    
    pc = f[timestamp]['lidar'][:]           # (N, 3)
    flow = f[timestamp]['flow_est_fixed'][:]  # (N, 3)
    ground = f[timestamp]['ground_mask'][:]   # (N,)
    
    # 動的物体の抽出（地面除外 & flow大きい点）
    dynamic = ~ground & (np.linalg.norm(flow, axis=1) > 0.1)
```

## 動的/静的の分離

エゴモーション補償後のflowで判定：

```python
import numpy as np

with h5py.File('01.h5', 'r') as f:
    t0, t1 = '000100', '000101'
    
    pc0 = f[t0]['lidar'][:]
    pose0 = f[t0]['pose'][:]
    pose1 = f[t1]['pose'][:]
    flow = f[t0]['flow_est_fixed'][:]
    ground = f[t0]['ground_mask'][:]
    
    # エゴモーション計算
    ego_pose = np.linalg.inv(pose1) @ pose0
    ego_flow = pc0 @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0
    
    # 物体自体の動き（エゴモーション除去）
    object_flow = flow - ego_flow
    
    # 動的判定（閾値例: 0.05m）
    is_dynamic = np.linalg.norm(object_flow, axis=1) > 0.05
    is_dynamic[ground] = False  # 地面は静的
```

## インデックスファイル

```python
import pickle

with open('index_total.pkl', 'rb') as f:
    index = pickle.load(f)
# index = [['01', '000000'], ['01', '000001'], ...]
```


