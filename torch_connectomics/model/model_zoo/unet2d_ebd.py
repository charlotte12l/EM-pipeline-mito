

import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class unet2d_ebd(nn.Module):

    def __init__(self):
        super().__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up10 = nn.Sequential(
                    nn.Conv2d(128+64, 64, 3, padding=1),
                    nn.ReLU(inplace=True))
        self.dconv_up11 = nn.Sequential(
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True))

        #self.proc_feats = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #self.proc_mul = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(64, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, x, prior):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up10(x)
        x = self.dconv_up11(x)

        x = self.conv_last(x)
        out = self.act(x)
        # c1 = self.proc_mul(x*prior).sum(3, keepdim=True).sum(2, keepdim=True)/(prior.sum())# foreground,becomes 128
        # c2 = self.proc_mul(x*(1-prior)).sum(3, keepdim=True).sum(2, keepdim=True)/((1-prior).sum())
        # feats = self.proc_feats(x) # 64->128
        # print(c1.shape)
        # print(c2.shape)
        # print(feats.shape)
        # dist1 = (feats - c1) ** 2
        # dist1 = torch.sqrt(dist1.sum(dim=1, keepdim=True))# sum channels
        # print(dist1.shape)
        # dist2 = (feats - c2) ** 2
        # dist2 = torch.sqrt(dist2.sum(dim=1, keepdim=True))
        # print(dist2.shape)
        return out


def test():
    model = unet2d_ebd()
    # print('model type: ', model.__class__.__name__)
    # num_params = sum([p.data.nelement() for p in model.parameters()])
    # print('number of trainable parameters: ', num_params)
    x = torch.randn(8, 1, 128, 128)
    label = torch.empty(8, 1, 128, 128).random_(2)
    y = model(x,label)
    print(x.size(), y.size())


if __name__ == '__main__':
    test()
