# Copyright (c) OpenMMLab. All rights reserved.
# Compatibility layer: expose voxelization and scatter_points from parent module
from ..voxelize import Voxelization, voxelization
from ..scatter_points import DynamicScatter, dynamic_scatter

__all__ = [
    'Voxelization', 'voxelization',
    'dynamic_scatter', 'DynamicScatter'
]

