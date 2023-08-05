# import os
# import requests
# from requests.adapters import HTTPAdapter
# import tempfile
# import torch
# from torch import nn
# from torch.nn import functional as F
# from urllib.request import urlopen, Request
# from tqdm.auto import tqdm
# import hashlib
# import shutil

# def download_url_to_file(url, dst, hash_prefix=None, progress=True):
#     file_size = None
#     req = Request(url, headers={"User-Agent": "torch.hub"})
#     u = urlopen(req)
#     meta = u.info()
#     if hasattr(meta, 'getheaders'):
#         content_length = meta.getheaders("Content-Length")
#     else:
#         content_length = meta.get_all("Content-Length")
#     if content_length is not None and len(content_length) > 0:
#         file_size = int(content_length[0])

#     dst = os.path.expanduser(dst)
#     dst_dir = os.path.dirname(dst)
#     f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

#     try:
#         if hash_prefix is not None:
#             sha256 = hashlib.sha256()
#         with tqdm(total=file_size, disable=not progress,
#                   unit='B', unit_scale=True, unit_divisor=1024) as pbar:
#             while True:
#                 buffer = u.read(8192)
#                 if len(buffer) == 0:
#                     break
#                 f.write(buffer)
#                 if hash_prefix is not None:
#                     sha256.update(buffer)
#                 pbar.update(len(buffer))

#         f.close()
#         if hash_prefix is not None:
#             digest = sha256.hexdigest()
#             if digest[:len(hash_prefix)] != hash_prefix:
#                 raise RuntimeError('invalid hash value (expected "{}", got "{}")'
#                                    .format(hash_prefix, digest))
#         shutil.move(f.name, dst)
#     finally:
#         f.close()
#         if os.path.exists(f.name):
#             os.remove(f.name)

# class BasicConv2d(nn.Module):

#     def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_planes, out_planes,
#             kernel_size=kernel_size, stride=stride,
#             padding=padding, bias=False
#         ) # verify bias false
#         self.bn = nn.BatchNorm2d(
#             out_planes,
#             eps=0.001, # value found in tensorflow
#             momentum=0.1, # default pytorch value
#             affine=True
#         )
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x


# class Block35(nn.Module):

#     def __init__(self, scale=1.0):
#         super().__init__()

#         self.scale = scale

#         self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

#         self.branch1 = nn.Sequential(
#             BasicConv2d(256, 32, kernel_size=1, stride=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         )

#         self.branch2 = nn.Sequential(
#             BasicConv2d(256, 32, kernel_size=1, stride=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         )

#         self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         out = torch.cat((x0, x1, x2), 1)
#         out = self.conv2d(out)
#         out = out * self.scale + x
#         out = self.relu(out)
#         return out


# class Block17(nn.Module):

#     def __init__(self, scale=1.0):
#         super().__init__()

#         self.scale = scale

#         self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

#         self.branch1 = nn.Sequential(
#             BasicConv2d(896, 128, kernel_size=1, stride=1),
#             BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
#             BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
#         )

#         self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
#         self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         out = torch.cat((x0, x1), 1)
#         out = self.conv2d(out)
#         out = out * self.scale + x
#         out = self.relu(out)
#         return out


# class Block8(nn.Module):

#     def __init__(self, scale=1.0, noReLU=False):
#         super().__init__()

#         self.scale = scale
#         self.noReLU = noReLU

#         self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

#         self.branch1 = nn.Sequential(
#             BasicConv2d(1792, 192, kernel_size=1, stride=1),
#             BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
#             BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
#         )

#         self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
#         if not self.noReLU:
#             self.relu = nn.ReLU(inplace=False)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         out = torch.cat((x0, x1), 1)
#         out = self.conv2d(out)
#         out = out * self.scale + x
#         if not self.noReLU:
#             out = self.relu(out)
#         return out


# class Mixed_6a(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

#         self.branch1 = nn.Sequential(
#             BasicConv2d(256, 192, kernel_size=1, stride=1),
#             BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(192, 256, kernel_size=3, stride=2)
#         )

#         self.branch2 = nn.MaxPool2d(3, stride=2)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         out = torch.cat((x0, x1, x2), 1)
#         return out


# class Mixed_7a(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.branch0 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 384, kernel_size=3, stride=2)
#         )

#         self.branch1 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=2)
#         )

#         self.branch2 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=2)
#         )

#         self.branch3 = nn.MaxPool2d(3, stride=2)

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#         out = torch.cat((x0, x1, x2, x3), 1)
#         return out


# class InceptionResnetV1(nn.Module):
#     def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
#         super().__init__()

#         # Set simple attributes
#         self.pretrained = pretrained
#         self.classify = classify
#         self.num_classes = num_classes

#         if pretrained == 'vggface2':
#             tmp_classes = 8631
#         elif pretrained == 'casia-webface':
#             tmp_classes = 10575
#         elif pretrained is None and self.classify and self.num_classes is None:
#             raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')
#         self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
#         self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
#         self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.maxpool_3a = nn.MaxPool2d(3, stride=2)
#         self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
#         self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
#         self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
#         self.repeat_1 = nn.Sequential(
#             Block35(scale=0.17),
#             Block35(scale=0.17),
#             Block35(scale=0.17),
#             Block35(scale=0.17),
#             Block35(scale=0.17),
#         )
#         self.mixed_6a = Mixed_6a()
#         self.repeat_2 = nn.Sequential(
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#             Block17(scale=0.10),
#         )
#         self.mixed_7a = Mixed_7a()
#         self.repeat_3 = nn.Sequential(
#             Block8(scale=0.20),
#             Block8(scale=0.20),
#             Block8(scale=0.20),
#             Block8(scale=0.20),
#             Block8(scale=0.20),
#         )
#         self.block8 = Block8(noReLU=True)
#         self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.last_linear = nn.Linear(1792, 512, bias=False)
#         self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

#         if pretrained is not None:
#             self.logits = nn.Linear(512, tmp_classes)
#             load_weights(self, pretrained)

#         if self.classify and self.num_classes is not None:
#             self.logits = nn.Linear(512, self.num_classes)

#         self.device = torch.device('cpu')
#         if device is not None:
#             self.device = device
#             self.to(device)

#     def forward(self, x):
#         x = self.conv2d_1a(x)
#         x = self.conv2d_2a(x)
#         x = self.conv2d_2b(x)
#         x = self.maxpool_3a(x)
#         x = self.conv2d_3b(x)
#         x = self.conv2d_4a(x)
#         x = self.conv2d_4b(x)
#         x = self.repeat_1(x)
#         x = self.mixed_6a(x)
#         x = self.repeat_2(x)
#         x = self.mixed_7a(x)
#         x = self.repeat_3(x)
#         x = self.block8(x)
#         x = self.avgpool_1a(x)
#         x = self.dropout(x)
#         x = self.last_linear(x.view(x.shape[0], -1))
#         x = self.last_bn(x)
#         if self.classify:
#             x = self.logits(x)
#         else:
#             x = F.normalize(x, p=2, dim=1)
#         return x


# def load_weights(mdl, name):
#     if name == 'vggface2':
#         path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
#     elif name == 'casia-webface':
#         path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
#     else:
#         raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

#     model_dir = os.path.join(get_torch_home(), 'checkpoints')
#     os.makedirs(model_dir, exist_ok=True)

#     cached_file = os.path.join(model_dir, os.path.basename(path))
#     if not os.path.exists(cached_file):
#         download_url_to_file(path, cached_file)

#     state_dict = torch.load(cached_file)
#     mdl.load_state_dict(state_dict)


# def get_torch_home():
#     torch_home = os.path.expanduser(
#         os.getenv(
#             'TORCH_HOME',
#             os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
#         )
#     )
#     return torch_home
import torch
from torch import nn
from torch import Tensor
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Callable, Any, Optional, List


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model