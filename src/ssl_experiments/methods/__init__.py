"""Self-supervised learning methods for detector representation learning."""

from ssl_experiments.methods.vicreg import (
    AlignedFeatureConsistencyLoss,
    MultiOutputVICRegModel,
    ObjectCentricDataset,
    SpatialVICRegLoss,
    VICRegLoss,
    VICRegModel,
    create_yolo_backbone,
    create_yolo_trunk,
    load_ssl_backbone,
)

__all__ = [
    "AlignedFeatureConsistencyLoss",
    "MultiOutputVICRegModel",
    "ObjectCentricDataset",
    "SpatialVICRegLoss",
    "VICRegLoss",
    "VICRegModel",
    "create_yolo_backbone",
    "create_yolo_trunk",
    "load_ssl_backbone",
]
