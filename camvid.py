import os
from PIL import Image
from collections import OrderedDict
from torchvision.datasets import vision, utils
from typing import Any, Callable, List, Optional, Tuple

class CamVid(vision.VisionDataset):
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])
    
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
        
        image_dir = os.path.join(root, f"{image_set}")
        target_dir = os.path.join(root, f"{image_set}annot")
        
        with open(os.path.join(root, f"{image_set}.txt"), 'r') as f:
            image_names, target_names = [], []
            for line in f:
                line = line.strip().split()
                image_names.append(line[0].split("/")[-1].strip())
                target_names.append(line[1].split("/")[-1].strip())
            
        self.images = [os.path.join(image_dir, x) for x in image_names]
        self.targets = [os.path.join(target_dir, x) for x in target_names]
        
        assert len(self.images) == len(self.targets)
    
    def __len__(self) -> int:
        return len(self.images)
    
    @property
    def masks(self) -> List[str]:
        return self.targets
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
        
if __name__ == "__main__":
    from utils import PILToLongTensor, LongTensorToRGBPIL, batch_transform, imshow_batch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((360, 480)),
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.Resize((360, 480)),
        PILToLongTensor()
    ])

    train_data = CamVid("/mnt/e/data/CamVid/SegNet-Tutorial/CamVid", "train", 
                        transform=transform, target_transform=target_transform)
    train_loader = DataLoader(train_data, 5)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_data.color_encoding

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Get a batch of samples to display
    images, labels = next(iter(train_loader))

    # Show a batch of samples and labels
    print("Close the figure window to continue...")
    label_to_rgb = transforms.Compose([
        LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_labels = batch_transform(labels, label_to_rgb)
    imshow_batch(images, color_labels)