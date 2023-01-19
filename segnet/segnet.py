from torch import nn, Tensor
from typing import List, Optional, Tuple
from torchvision.models import vgg16, VGG16_Weights

__all__ = ["SegNet", "BayesSegNet"]

def _make_layer(
    in_channels: int, 
    out_channels: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class EncBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_layers: int
    ) -> None:
        super().__init__()
        layers = [_make_layer(in_channels, out_channels)]
        for _ in range(num_layers - 1):
            layers += [_make_layer(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.layers(x)
        x, indices = self.max_pool(x)
        return x, indices
    
class BayesEncBlock(EncBlock):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_layers: int
    ) -> None:
        super().__init__(in_channels, out_channels, num_layers)
        self.dropout = nn.Dropout(0.5, inplace=False)
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x, indices = super().forward(x)
        x = self.dropout(x)
        return x, indices
    
class DecBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_layers: int
    ) -> None:
        super().__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        layers = []
        for _ in range(num_layers - 1):
            layers += [_make_layer(in_channels, in_channels)]
        layers += [_make_layer(in_channels, out_channels)]
        self.layers = nn.Sequential(*layers)
        
    def forward(
        self, x: Tensor, 
        indices: Tensor, 
        output_size: Optional[List[int]] = None
    ) -> Tensor:
        x = self.max_unpool(x, indices, output_size)
        x = self.layers(x)
        return x
    
class BayesDecBlock(DecBlock):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_layers: int
    ) -> None:
        super().__init__(in_channels, out_channels, num_layers)
        self.dropout = nn.Dropout(0.5, inplace=False)
        
    def forward(
        self, 
        x: Tensor, 
        indices: Tensor, 
        output_size: Optional[List[int]] = None
    ) -> Tensor:
        x = super().forward(x, indices, output_size)
        x = self.dropout(x)
        return x

class SegNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, vgg_encoder: bool = True) -> None:
        super().__init__()
        self.encoder0 = EncBlock(in_channels, 64, 2)
        self.encoder1 = EncBlock(64, 128, 2)
        self.encoder2 = EncBlock(128, 256, 3)
        self.encoder3 = EncBlock(256, 512, 3)
        self.encoder4 = EncBlock(512, 512, 3)
        
        self.decoder4 = DecBlock(512, 512, 3)
        self.decoder3 = DecBlock(512, 256, 3)
        self.decoder2 = DecBlock(256, 128, 3)
        self.decoder1 = DecBlock(128, 64, 2)
        self.decoder0 = DecBlock(64, 64, 1)
        
        self.conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.softmax = nn.Softmax2d()
        
        if vgg_encoder:
            self._init_vgg16_encoder()
    
    def _init_vgg16_encoder(self):
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        params = []
        for module in vgg.modules():
            if isinstance(module, nn.Conv2d):
                params += [module.state_dict()]
        
        idx = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and name.startswith("encoder"):
                module.load_state_dict(params[idx])
                idx += 1
        
    def forward(self, x: Tensor) -> Tensor:
        dim0 = x.size()
        x, indices0 = self.encoder0(x)
        dim1 = x.size()
        x, indices1 = self.encoder1(x)
        dim2 = x.size()
        x, indices2 = self.encoder2(x)
        dim3 = x.size()
        x, indices3 = self.encoder3(x)
        dim4 = x.size()
        x, indices4 = self.encoder4(x)
        
        x = self.decoder4(x, indices4, dim4)
        x = self.decoder3(x, indices3, dim3)
        x = self.decoder2(x, indices2, dim2)
        x = self.decoder1(x, indices1, dim1)
        x = self.decoder0(x, indices0, dim0)
        x = self.conv(x)
        x = self.softmax(x)
        return x
        
class BayesSegNet(SegNet):
    def __init__(self, in_channels: int, out_channels: int, vgg_encoder: bool = True) -> None:
        super().__init__(in_channels, out_channels, False)
        self.encoder2 = BayesEncBlock(128, 256, 3)
        self.encoder3 = BayesEncBlock(256, 512, 3)
        self.encoder4 = BayesEncBlock(512, 512, 3)
        
        self.decoder4 = BayesDecBlock(512, 512, 3)
        self.decoder3 = BayesDecBlock(512, 256, 3)
        self.decoder2 = BayesDecBlock(256, 128, 3)
        
        if vgg_encoder:
            self._init_vgg16_encoder()
    