import torch
import torch.nn as nn
import torch.nn.functional as F


# python2
# from layers import *
# python3
from .layers import *


class FCDenseNet(nn.Module):
    def __init__(self, in_channels, conv1_out_ch, down_blocks, bottleneck_layers, up_blocks, 
                    growth_rate, n_classes):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        skip_channels = []

        #####################
        # Conv1 #
        #####################
        self.conv1 = nn.Conv2d(in_channels, conv1_out_ch, 3, 1, 1)
        cur_channels_count = conv1_out_ch

        #####################
        # Downsampling path #
        #####################
        dense_down_blocks = []
        trans_down_blocks = []
        for ii, layer_num in enumerate(down_blocks):
            dense_down_blocks.append(
                DenseBlock(cur_channels_count, growth_rate, layer_num))
            cur_channels_count += (growth_rate*layer_num)
            skip_channels.insert(0, cur_channels_count)
            if ii != 4:
                trans_down_blocks.append(TransitionDown(cur_channels_count))
        self.denseDownBlocks = nn.ModuleList(dense_down_blocks)
        self.transDownBlocks = nn.ModuleList(trans_down_blocks)

        #######################
        #   top2bottom   #
        #######################
        # Lateral layers
        self.t2b_conv5 = nn.Conv2d(skip_channels[0], 128, 1, 1)
        self.t2b_conv4 = nn.Conv2d(skip_channels[1], 128, 1, 1)
        self.t2b_conv3 = nn.Conv2d(skip_channels[2], 128, 1, 1)
        self.t2b_conv2 = nn.Conv2d(skip_channels[3], 128, 1, 1)
        self.t2b_conv1 = nn.Conv2d(skip_channels[4], 128, 1, 1)

        # Smooth layers
        self.t2b_conv54 = nn.Conv2d(128, 128, 3, padding=1)
        self.t2b_conv43 = nn.Conv2d(128, 128, 3, padding=1)
        self.t2b_conv32 = nn.Conv2d(128, 128, 3, padding=1)
        self.t2b_conv21 = nn.Conv2d(128, 128, 3, padding=1)

        #######################
        #   bottom2top   #
        ####################### 
        # Lateral layers
        self.b2t_conv1 = nn.Conv2d(skip_channels[4], 128, 1, 1)
        self.b2t_conv2 = nn.Conv2d(skip_channels[3], 128, 1, 1)
        self.b2t_conv3 = nn.Conv2d(skip_channels[2], 128, 1, 1)
        self.b2t_conv4 = nn.Conv2d(skip_channels[1], 128, 1, 1)
        self.b2t_conv5 = nn.Conv2d(skip_channels[0], 128, 1, 1)

        # Smooth layers
        self.b2t_conv12 = nn.Conv2d(128, 128, 3, padding=1)
        self.b2t_conv23 = nn.Conv2d(128, 128, 3, padding=1)
        self.b2t_conv34 = nn.Conv2d(128, 128, 3, padding=1)
        self.b2t_conv45 = nn.Conv2d(128, 128, 3, padding=1)

        # Score layers
        self.t2b_score5 = nn.Sequential(
                        nn.Conv2d(skip_channels[0], skip_channels[0]//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(skip_channels[0]//4, 1, 1),
        )
        self.t2b_score4 = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1),
        )
        self.t2b_score3 = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1)
        )
        self.t2b_score2 = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1),
        )
        self.t2b_score1 = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1),
        )

        # Score layers
        self.b2t_score1 = nn.Sequential(
                        nn.Conv2d(skip_channels[4], skip_channels[4]//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(skip_channels[4]//4, 1, 1),
        )
        self.b2t_score2 = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1),
        )
        self.b2t_score3 = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1),
        )
        self.b2t_score4 = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1),
        )
        self.b2t_score5 = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1),
        )

        #######################
        #   attention fuse   #
        #######################
        self.att_fuse = nn.Sequential(
                        nn.Conv2d(128*2, (128*2)//8, 3, padding=1),
                        nn.LeakyReLU(0.02),
                        nn.Conv2d((128*2)//8, 2, 1),
                        nn.Sigmoid(),
        )

        # final score
        self.score = nn.Sequential(
                        nn.Conv2d(128, 128//4, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128//4, 1, 1),
        )

        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        _bsz, _c, _in_H, _in_W = x.size()
        x = self.conv1(x)

        xs = []
        for i in range(len(self.down_blocks)):
            x = self.denseDownBlocks[i](x)
            # print("down", i, x.size()) 
            if i != 4:
                xs.append(x)
                x = self.transDownBlocks[i](x)

        # Top-down
        _x = self.t2b_score5(x)
        _x = F.interpolate(_x, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        t2b_y5 = self.sigmoid(_x)

        p5 = self.t2b_conv5(x)
        p4 = self._upsample_add(p5, self.t2b_conv4(xs[3]) )
        p4 = self.t2b_conv54(p4)
        _p4 = self.t2b_score4(p4)
        _p4 = F.interpolate(_p4, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        t2b_y4 = self.sigmoid(_p4)

        p3 = self._upsample_add(p4, self.t2b_conv3(xs[2]) )
        p3 = self.t2b_conv43(p3)
        _p3 = self.t2b_score3(p3)
        _p3 = F.interpolate(_p3, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        t2b_y3 = self.sigmoid(_p3)

        p2 = self._upsample_add(p3, self.t2b_conv2(xs[1]) )
        p2 = self.t2b_conv32(p2)
        _p2 = self.t2b_score2(p2)
        _p2 = F.interpolate(_p2, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        t2b_y2 = self.sigmoid(_p2)

        p1 = self._upsample_add(p2, self.t2b_conv1(xs[0]) )
        p1 = self.t2b_conv21(p1)
        _p1 = self.t2b_score1(p1)
        t2b_y1 = self.sigmoid(_p1)

        # Bottom-top
        _b2t_y1 = self.b2t_score1(xs[0])
        b2t_y1 = self.sigmoid(_b2t_y1)

        b2t_p1 = self.b2t_conv1(xs[0])
        b2t_p2 = self._upsample_add(self.b2t_conv2(xs[1]), b2t_p1)
        b2t_p2 = self.b2t_conv12(b2t_p2)
        _b2t_p2 = self.b2t_score2(b2t_p2)
        _b2t_p2 = F.interpolate(_b2t_p2, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        b2t_y2 = self.sigmoid(_b2t_p2)

        b2t_p3 = self._upsample_add(self.b2t_conv3(xs[2]), b2t_p2)
        b2t_p3 = self.b2t_conv23(b2t_p3)
        _b2t_p3 = self.b2t_score3(b2t_p3)
        _b2t_p3 = F.interpolate(_b2t_p3, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        b2t_y3 = self.sigmoid(_b2t_p3)

        b2t_p4 = self._upsample_add(self.b2t_conv4(xs[3]), b2t_p3)
        b2t_p4 = self.b2t_conv34(b2t_p4)
        _b2t_p4 = self.b2t_score4(b2t_p4)
        _b2t_p4 = F.interpolate(_b2t_p4, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        b2t_y4 = self.sigmoid(_b2t_p4)

        b2t_p5 = self._upsample_add(self.b2t_conv5(x), b2t_p4)
        b2t_p5 = self.b2t_conv45(b2t_p5)
        _b2t_p5 = self.b2t_score5(b2t_p5)
        _b2t_p5 = F.interpolate(_b2t_p5, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        b2t_y5 = self.sigmoid(_b2t_p5)

        # attention fuse
        b2t_p5 = F.interpolate(b2t_p5, size=(_in_H, _in_W), mode='bilinear', align_corners=True)
        att_maps = self.att_fuse(torch.cat((b2t_p5, p1), dim=1))
        _fea = att_maps[:, 0].unsqueeze(1).contiguous() * b2t_p5 + \
                att_maps[:, 1].unsqueeze(1).contiguous() * p1

        y = self.score(_fea)
        y = self.sigmoid(y)
        
        if self.training:
            return [y, t2b_y5, t2b_y4, t2b_y3, t2b_y2, t2b_y1, b2t_y1, b2t_y2, b2t_y3, b2t_y4, b2t_y5]
        else:
            return y


def FCDenseNet103(n_classes):
    return FCDenseNet(3, 48, (4, 5, 7, 10, 12), 15, (12, 10, 7, 5, 4), 16, n_classes)
