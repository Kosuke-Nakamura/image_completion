#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

 
# 補完ネットワーク
class CompletionNet(chainer.Chain):

    def __init__(self):
        super(CompletionNet, self).__init__(
            # とりあえず初期化はdefaultで
            c0_0 = L.Convolution2D(4, 64, ksize=5, stride=1, pad=2),
            c0_1 = L.Convolution2D(64, 128, ksize=3, stride=2, pad=1),  # ダウンスケーリング
            c0_2 = L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),
            c0_3 = L.Convolution2D(128, 256, ksize=3, stride=2, pad=1), # ダウンスケーリング
            c0_4 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            c0_5 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            # Dilated Convolution (padは出力サイズが変わらないように設定)
            dlc0 = L.DilatedConvolution2D(256, 256, ksize=3, stride=1, pad=2, dilate=2),
            dlc1 = L.DilatedConvolution2D(256, 256, ksize=3, stride=1, pad=4, dilate=4),
            dlc2 = L.DilatedConvolution2D(256, 256, ksize=3, stride=1, pad=8, dilate=8),
            dlc3 = L.DilatedConvolution2D(256, 256, ksize=3, stride=1, pad=16, dilate=16),
            # アップスケーリング
            c1_0 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            c1_1 = L.Convolution2D(256, 256, ksize=3, stride=1, pad=1),
            dec0 = L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1), # アップスケーリング
            c1_2 = L.Convolution2D(128, 128, ksize=3, stride=1, pad=1),
            dec1 = L.Deconvolution2D(128, 64, ksize=4, stride=2, pad=1),  # アップスケーリング
            c1_3 = L.Convolution2D(64, 32, ksize=3, stride=1, pad=1),
            c1_4 = L.Convolution2D(32, 3, ksize=3, stride=1, pad=1), # 不明だが他の層と統一するということでpad=1
            # Batch Normalization
            bn00 = L.BatchNormalization(64),
            bn01 = L.BatchNormalization(128),
            bn02 = L.BatchNormalization(128),
            bn03 = L.BatchNormalization(256),
            bn04 = L.BatchNormalization(256),
            bn05 = L.BatchNormalization(256),
            bn06 = L.BatchNormalization(256),
            bn07 = L.BatchNormalization(256),
            bn08 = L.BatchNormalization(256),
            bn09 = L.BatchNormalization(256),
            bn10 = L.BatchNormalization(256),
            bn11 = L.BatchNormalization(256),
            bn12 = L.BatchNormalization(128),
            bn13 = L.BatchNormalization(128),
            bn14 = L.BatchNormalization(64),
            bn15 = L.BatchNormalization(32))
            

    def __call__(self, z):
        h = F.relu(self.bn00(self.c0_0(z)))
        h = F.relu(self.bn01(self.c0_1(h)))
        h = F.relu(self.bn02(self.c0_2(h)))
        h = F.relu(self.bn03(self.c0_3(h)))
        h = F.relu(self.bn04(self.c0_4(h)))
        h = F.relu(self.bn05(self.c0_5(h)))
        h = F.relu(self.bn06(self.dlc0(h)))
        h = F.relu(self.bn07(self.dlc1(h)))
        h = F.relu(self.bn08(self.dlc2(h)))
        h = F.relu(self.bn09(self.dlc3(h)))
        h = F.relu(self.bn10(self.c1_0(h)))
        h = F.relu(self.bn11(self.c1_1(h)))
        h = F.relu(self.bn12(self.dec0(h)))
        h = F.relu(self.bn13(self.c1_2(h)))
        h = F.relu(self.bn14(self.dec1(h)))
        h = F.relu(self.bn15(self.c1_3(h)))
        x = F.sigmoid(self.c1_4(h))
        
        return x

# 識別ネットワーク
class Discriminator(chainer.Chain):

    def __init__(self, global_size=256, local_size=128):
        l_size = local_size
        g_size = global_size

        # convolution による 画像サイズの変化を計算
        for i in range(5):
            l_size = (l_size - 1) // 2 + 1
        for i in range(6):
            g_size = (g_size - 1) // 2 + 1
            
        super(Discriminator, self).__init__(
            # Local Discriminator
            ld_c0 = L.Convolution2D(3, 64, ksize=5, stride=2, pad=2),
            ld_c1 = L.Convolution2D(64, 128, ksize=5, stride=2, pad=2),
            ld_c2 = L.Convolution2D(128, 256, ksize=5, stride=2, pad=2),
            ld_c3 = L.Convolution2D(256, 512, ksize=5, stride=2, pad=2),
            ld_c4 = L.Convolution2D(512, 512, ksize=5, stride=2, pad=2),
            ld_fc = L.Linear(512 * l_size * l_size, 1024),
            # Golobal Discriminator
            gd_c0 = L.Convolution2D(3, 64, ksize=5, stride=2, pad=2),
            gd_c1 = L.Convolution2D(64, 128, ksize=5, stride=2, pad=2),
            gd_c2 = L.Convolution2D(128, 256, ksize=5, stride=2, pad=2),
            gd_c3 = L.Convolution2D(256, 512, ksize=5, stride=2, pad=2),
            gd_c4 = L.Convolution2D(512, 512, ksize=5, stride=2, pad=2),
            gd_c5 = L.Convolution2D(512, 512, ksize=5, stride=2, pad=2),
            gd_fc = L.Linear(512 * g_size * g_size, 1024),
            # Concatenation layer
            concl = L.Linear(2048, 1),
            # Batch Normalization
            ld_bn0 = L.BatchNormalization(64),
            ld_bn1 = L.BatchNormalization(128),
            ld_bn2 = L.BatchNormalization(256),
            ld_bn3 = L.BatchNormalization(512),
            ld_bn4 = L.BatchNormalization(512),
            gd_bn0 = L.BatchNormalization(64),
            gd_bn1 = L.BatchNormalization(128),
            gd_bn2 = L.BatchNormalization(256),
            gd_bn3 = L.BatchNormalization(512),
            gd_bn4 = L.BatchNormalization(512),
            gd_bn5 = L.BatchNormalization(512))
            
            
    def __call__(self, x_ld, x_gd):
        # Local Discriminator
        hl = F.leaky_relu(self.ld_bn0(self.ld_c0(x_ld)))
        hl = F.leaky_relu(self.ld_bn1(self.ld_c1(hl)))
        hl = F.leaky_relu(self.ld_bn2(self.ld_c2(hl)))
        hl = F.leaky_relu(self.ld_bn3(self.ld_c3(hl)))
        hl = F.leaky_relu(self.ld_bn4(self.ld_c4(hl)))
        hl = F.leaky_relu(self.ld_fc(hl))
        # Global Discriminator
        hg = F.leaky_relu(self.gd_bn0(self.gd_c0(x_gd)))
        hg = F.leaky_relu(self.gd_bn1(self.gd_c1(hg)))
        hg = F.leaky_relu(self.gd_bn2(self.gd_c2(hg)))
        hg = F.leaky_relu(self.gd_bn3(self.gd_c3(hg)))
        hg = F.leaky_relu(self.gd_bn4(self.gd_c4(hg)))
        hg = F.leaky_relu(self.gd_bn5(self.gd_c5(hg)))
        hg = F.leaky_relu(self.gd_fc(hg))
        # concatenation
        out = F.concat((hl, hg), axis=1)
        out = self.concl(out)
        return out
