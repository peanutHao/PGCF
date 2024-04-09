from lib.pvtv2 import pvt_v2_b2
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Conv2d, UpsamplingBilinear2d
import warnings
import torch
from mmcv.cnn import constant_init, kaiming_init
from torch import nn

warnings.filterwarnings('ignore')


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class InceptionDilatedConv2d(nn.Module):

    def __init__(self, in_channels, branch_ratio=0.25):
        super().__init__()
        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels - 3 * gc, in_channels - 3 * gc, 1, bias=False),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True))
        self.conv2 = ASPPConv(gc, gc, 6)
        self.conv3 = ASPPConv(gc, gc, 12)
        self.conv4 = ASPPConv(gc, gc, 18)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        self.pool = ASPPPooling(in_channels, in_channels)
        self.pool1 = ASPPPooling(in_channels, in_channels // 2)
        self.pool2 = ASPPPooling(in_channels, in_channels // 2)
        self.project = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, 1, bias=False),  # 卷积核1。
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

        self.conv_2_1 = nn.Conv2d(gc * 4, gc * 2, 1, 1)
        self.conv_2_2 = nn.Conv2d(gc * 4, gc * 2, 1, 1)

    def forward(self, x):
        x_1, x_2, x_3, x_4 = torch.split(x, self.split_indexes, dim=1)
        X2_1 = torch.cat((self.conv1(x_1), self.conv2(x_2)), dim=1)
        X2_2 = torch.cat((self.conv3(x_3), self.conv4(x_4)), dim=1)
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        X2_1 = torch.cat((X2_1, pool1), dim=1)
        X2_2 = torch.cat((X2_2, pool2), dim=1)
        y = torch.cat((self.conv_2_1(X2_1), self.conv_2_2(X2_2)), dim=1)
        y = torch.cat(
            (y, self.pool(x)),
            dim=1,
        )
        y = torch.add(x, self.project(y))
        return y


def Upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class PGCF(nn.Module):
    def __init__(self, dim=32, dims=[64, 128, 320, 512]):
        super(PGCF, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/home/users/qianhao/PycharmProjects/PGCF//pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]

        self.InDD_c4 = InceptionDilatedConv2d(c4_in_channels)
        self.InDD_c3 = InceptionDilatedConv2d(c3_in_channels)
        self.InDD_c2 = InceptionDilatedConv2d(c2_in_channels)
        self.InDD_c1 = InceptionDilatedConv2d(c1_in_channels)
        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=dim)
        self.linear_fuse = ConvModule(in_channels=dim * 4, out_channels=dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = Conv2d(dim, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        self.linear_pred1 = Conv2d(dim, 1, kernel_size=1)
        self.dropout1 = nn.Dropout(0.1)
        self.linear_pred2 = Conv2d(dim, 1, kernel_size=1)
        self.dropout2 = nn.Dropout(0.1)
        self.deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.linear_pred_f = Conv2d(3, 1, kernel_size=1)
        self.ca11 = ChannelAttention(in_planes=dim)
        self.ca21 = ChannelAttention(in_planes=dim)
        self.ca31 = ChannelAttention(in_planes=dim)
        self.sa11 = SpatialAttention(kernel_size=3)
        self.sa21 = SpatialAttention(kernel_size=3)
        self.sa31 = SpatialAttention(kernel_size=7)

        self.ca1 = ChannelAttention(in_planes=c1_in_channels)
        self.ca2 = ChannelAttention(in_planes=c2_in_channels)
        self.ca3 = ChannelAttention(in_planes=c3_in_channels)
        self.ca4 = ChannelAttention(in_planes=c4_in_channels)

        self.sa1 = SpatialAttention(kernel_size=3)
        self.sa2 = SpatialAttention(kernel_size=3)
        self.sa3 = SpatialAttention(kernel_size=7)
        self.sa4 = SpatialAttention(kernel_size=7)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        c1, c2, c3, c4 = pvt
        n, _, h, w = c4.shape
        _c4 = self.InDD_c4(c4)  # [1, 64, 11, 11]
        _c3 = self.InDD_c3(c3)  # [1, 64, 22, 22]
        _c2 = self.InDD_c2(c2)  # [1, 64, 44, 44]
        _c1 = self.InDD_c1(c1)

        _c4 = self.sa4(self.ca4(_c4)) * _c4
        _c3 = self.sa3(self.ca3(_c3)) * _c3
        _c2 = self.sa2(self.ca2(_c2)) * _c2
        _c1 = self.sa1(self.ca1(_c1)) * _c1

        _c4 = self.linear_c4(_c4).permute(0, 2, 1).reshape(n, -1, _c4.shape[2], _c4.shape[3])
        _c3 = self.linear_c3(_c3).permute(0, 2, 1).reshape(n, -1, _c3.shape[2], _c3.shape[3])
        _c2 = self.linear_c2(_c2).permute(0, 2, 1).reshape(n, -1, _c2.shape[2], _c2.shape[3])
        _c1 = self.linear_c1(_c1).permute(0, 2, 1).reshape(n, -1, _c1.shape[2], _c1.shape[3])

        _c4 = resize(_c4, size=_c3.size()[2:], mode='bilinear', align_corners=False)
        sub = abs(_c3 - _c4)
        _fu1 = self.sa11(self.ca11(torch.add(_c4, _c3)))
        _c31 = _fu1 * _c3
        _c41 = _fu1 * _c4
        L34 = self.linear_fuse34(torch.add(torch.add(_c41, _c31), sub))
        O34 = L34

        _c3 = resize(_c3, size=_c2.size()[2:], mode='bilinear', align_corners=False)
        sub = abs(_c2 - _c3)
        L34 = resize(L34, size=_c2.size()[2:], mode='bilinear', align_corners=False)
        _fu2 = self.sa21(self.ca21(torch.add(L34, _c2)))
        _c21 = _fu2 * _c2
        L341 = _fu2 * L34
        L2 = self.linear_fuse2(torch.add(torch.add(L341, _c21), sub))
        O2 = L2

        _c2 = resize(_c2, size=_c1.size()[2:], mode='bilinear', align_corners=False)
        sub = abs(_c1 - _c2)
        L2 = resize(L2, size=_c1.size()[2:], mode='bilinear', align_corners=False)
        _fu3 = self.sa31(self.ca31(torch.add(L2, _c1)))
        _c11 = _fu3 * _c1
        L21 = _fu3 * L2
        _c = self.linear_fuse1(torch.add(torch.add(L21, _c11), sub))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # x = self.deconv(x)
        O2 = self.dropout2(O2)
        O2 = self.linear_pred2(O2)
        O34 = self.dropout1(O34)
        O34 = self.linear_pred1(O34)
        return x, O2, O34





