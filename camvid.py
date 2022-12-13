from torchvision.datasets import vision, utils
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image

class CamVid(vision.VisionDataset):
    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        valid_image_sets = ["train", "val", "test"]
        self.image_set = utils.verify_str_arg(image_set, "image_set", valid_image_sets)
        