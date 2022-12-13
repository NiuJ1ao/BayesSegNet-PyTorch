from torchvision.datasets import VOCSegmentation, VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms

batch_size = 100
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_data = VOCSegmentation("/mnt/e/data/PascalVOC/", image_set="trainval", download=True, transform=transform, target_transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print(len(train_loader.dataset))
for x, y in train_loader:
    print(x.shape, y.shape)
