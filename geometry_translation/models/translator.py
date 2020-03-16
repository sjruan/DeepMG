import torch
import torch.nn as nn


class T2RNet_Trans(nn.Module):
    def __init__(self, input_nc, base_channels):
        super(T2RNet_Trans, self).__init__()
        self.max_pool = nn.MaxPool2d(2)

        self.conv1 = self.conv_stage(input_nc, base_channels)
        self.conv2 = self.conv_stage(base_channels, base_channels * 2)
        self.conv3 = self.conv_stage(base_channels * 2, base_channels * 4)
        self.conv4 = self.conv_stage(base_channels * 4, base_channels * 8)

        self.center = self.conv_stage(base_channels * 8, base_channels * 16)

        self.d1_conv4 = self.conv_stage(base_channels * 16, base_channels * 8)
        self.d1_conv3 = self.conv_stage(base_channels * 8, base_channels * 4)
        self.d1_conv2 = self.conv_stage(base_channels * 4, base_channels * 2)
        self.d1_conv1 = self.conv_stage(base_channels * 2, base_channels)

        self.d1_up4 = self.upsample(base_channels * 16, base_channels * 8)
        self.d1_up3 = self.upsample(base_channels * 8, base_channels * 4)
        self.d1_up2 = self.upsample(base_channels * 4, base_channels * 2)
        self.d1_up1 = self.upsample(base_channels * 2, base_channels)

        self.d1_conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.d2_conv4 = self.conv_stage(base_channels * (16 + 8), base_channels * 8)
        self.d2_conv3 = self.conv_stage(base_channels * (8 + 4), base_channels * 4)
        self.d2_conv2 = self.conv_stage(base_channels * (4 + 2), base_channels * 2)
        self.d2_conv1 = self.conv_stage(base_channels * (2 + 1), base_channels)

        self.d2_up4 = self.upsample(base_channels * 16, base_channels * 8)
        self.d2_up3 = self.upsample(base_channels * 8, base_channels * 4)
        self.d2_up2 = self.upsample(base_channels * 4, base_channels * 2)
        self.d2_up1 = self.upsample(base_channels * 2, base_channels)

        self.d2_conv_last = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )

    def forward(self, input):
        # conv1_out: 256x256x64
        conv1_out = self.conv1(input)
        # 128 x 128
        # conv2_out: 128x128x128
        conv2_out = self.conv2(self.max_pool(conv1_out))
        # 64 x 64
        # conv3_out: 64x64x256
        conv3_out = self.conv3(self.max_pool(conv2_out))
        # 32 * 32
        # conv4_out: 32x32x512
        conv4_out = self.conv4(self.max_pool(conv3_out))
        # 16 * 16
        # out: 16x16x1024
        out = self.center(self.max_pool(conv4_out))

        # d1_up4_out: 32 * 32 * 512
        d1_up4_out = self.d1_up4(out)
        out1 = self.d1_conv4(torch.cat((d1_up4_out, conv4_out), 1))
        d1_up3_out = self.d1_up3(out1)
        out1 = self.d1_conv3(torch.cat((d1_up3_out, conv3_out), 1))
        d1_up2_out = self.d1_up2(out1)
        out1 = self.d1_conv2(torch.cat((d1_up2_out, conv2_out), 1))
        d1_up1_out = self.d1_up1(out1)
        out1 = self.d1_conv1(torch.cat((d1_up1_out, conv1_out), 1))
        out1 = self.d1_conv_last(out1)

        out2 = self.d2_conv4(torch.cat((self.d2_up4(out), conv4_out, d1_up4_out), 1))
        out2 = self.d2_conv3(torch.cat((self.d2_up3(out2), conv3_out, d1_up3_out), 1))
        out2 = self.d2_conv2(torch.cat((self.d2_up2(out2), conv2_out, d1_up2_out), 1))
        out2 = self.d2_conv1(torch.cat((self.d2_up1(out2), conv1_out, d1_up1_out), 1))
        out2 = self.d2_conv_last(out2)
        return out1, out2
