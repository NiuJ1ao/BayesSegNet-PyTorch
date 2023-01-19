import sys
sys.path.append("./bayesianize/")
import bnn
from .segnet import SegNet

__all__ = ["MFVISegNet"]

class MFVISegNet(SegNet):
    def __init__(self, in_channels: int, out_channels: int, vgg_encoder: bool = True) -> None:
        super().__init__(in_channels, out_channels, vgg_encoder)
        bnn.bayesianize_(self, inference="ffg", init_sd=1e-4, max_sd=None, reference_state_dict=self.state_dict())
