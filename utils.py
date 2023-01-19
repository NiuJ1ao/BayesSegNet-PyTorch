import torch
from torch import nn
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision.transforms import ToPILImage
    
def to_numpy(x):
    return x.detach().cpu().numpy()

class LocalContrastNormalisation(object):
    def __init__(self, kernel_size=3, mean=0, std=1):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1, bias=False)
        weight_size = self.conv2d.weight.size()
        kernel = torch.normal(mean=mean, std=std, size=weight_size)
        kernel /= torch.sum(kernel) # normalise kernel
        self.conv2d.weight = nn.Parameter(kernel, requires_grad=False)
        
    def __call__(self, x):
        v = x - torch.sum(torch.vstack([self.conv2d(i.unsqueeze(0)) for i in x]), dim=0, keepdim=True)
        sigma = torch.sum(torch.vstack([self.conv2d(torch.square(i).unsqueeze(0)) for i in v]), dim=0).sqrt()
        c = torch.mean(sigma)
        y = v / torch.maximum(sigma, c)
        assert y.size() == x.size()
        return y

class PILToLongTensor(object):
    """https://github.com/davidtvs/PyTorch-ENet/
    
    Converts a ``PIL Image`` to a ``torch.LongTensor``.
    Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor
    """

    def __call__(self, pic):
        """Performs the conversion from a ``PIL Image`` to a ``torch.LongTensor``.
        Keyword arguments:
        - pic (``PIL.Image``): the image to convert to ``torch.LongTensor``
        Returns:
        A ``torch.LongTensor``.
        """
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))

        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()

        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()

class LongTensorToRGBPIL(object):
    """https://github.com/davidtvs/PyTorch-ENet/
    
    Converts a ``torch.LongTensor`` to a ``PIL image``.
    The input is a ``torch.LongTensor`` where each pixel's value identifies the
    class.
    Keyword arguments:
    - rgb_encoding (``OrderedDict``): An ``OrderedDict`` that relates pixel
    values, class names, and class colors.
    """
    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """Performs the conversion from ``torch.LongTensor`` to a ``PIL image``
        Keyword arguments:
        - tensor (``torch.LongTensor``): the tensor to convert
        Returns:
        A ``PIL.Image``.
        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))

        # label_tensor might be an image without a channel dimension, in this
        # case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)

        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

        for index, (_, color) in enumerate(self.rgb_encoding.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)

        return ToPILImage()(color_tensor)
    
def batch_transform(batch, transform):
    """https://github.com/davidtvs/PyTorch-ENet/
    
    Applies a transform to a batch of samples.
    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``
    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    """https://github.com/davidtvs/PyTorch-ENet/
    
    Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``
    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()

def median_freq_balancing(dataloader, num_classes, device):
    """https://github.com/davidtvs/PyTorch-ENet/
    
    Computes class weights using median frequency balancing as described
    in https://arxiv.org/abs/1411.4734:
        w_class = median_freq / freq_class,
    where freq_class is the number of pixels of a given class divided by
    the total number of pixels in images where that class is present, and
    median_freq is the median of freq_class.
    Keyword arguments:
    - dataloader (``data.Dataloader``): A data loader to iterate over the
    dataset.
    whose weights are going to be computed.
    - num_classes (``int``): The number of classes
    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # Flatten label
        flat_label = label.flatten()

        # Sum up the class frequencies
        bincount = np.bincount(flat_label, minlength=num_classes)

        # Create of mask of classes that exist in the label
        mask = bincount > 0
        # Multiply the mask by the pixel count. The resulting array has
        # one element for each class. The value is either 0 (if the class
        # does not exist in the label) or equal to the pixel count (if
        # the class exists in the label)
        total += mask * flat_label.size

        # Sum up the number of pixels found for each class
        class_count += bincount

    # Compute the frequency and its median
    freq = class_count / total
    med = np.median(freq)

    return torch.tensor(med / freq, dtype=torch.float, device=device)
