from .deflow import DeFlow
from .fastflow3d import FastFlow3D
from .voteflow import VoteFlow

# NOTE: Older checkpoints (e.g., sf_voxel_model) reference SFVoxelModel.
# Provide a thin alias to keep backward compatibility with saved hyperparameters.
class SFVoxelModel(VoteFlow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)