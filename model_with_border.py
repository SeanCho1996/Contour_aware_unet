from unet2d import UNet
import torch
import torch.nn as nn


class BorderUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.unet = UNet(n_class)

        self.border_extraction = nn.Conv2d(64, 2, kernel_size=1, padding=0)

        # self.border_enhance = nn.Sequential(
        #     nn.Conv2d(2, 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(2, 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )

        self.fine_correction = nn.Sequential(
            nn.Conv2d(66, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, n_class, kernel_size=3, padding=1),
        )

        self.softmax = nn.Softmax2d()
    
    def forward(self, in_feat):
        # regular cnn process
        init_seg, unet_feature = self.unet(in_feat)

        # extract and enhance border
        init_border = self.border_extraction(unet_feature)
        # fine_border = self.border_enhance(init_border)

        # detach border so that backward won't affect border
        # fine_border_isolate = border.type(torch.float32)

        # fusion
        # fine_feature = torch.add(unet_feature, fine_border)
        # fine_feature = torch.add(unet_feature, fine_border_isolate)
        # fine_feature = torch.cat((unet_feature, fine_border_isolate), dim=1)

        
        # output
        out = self.softmax(init_seg)

        return out, init_border