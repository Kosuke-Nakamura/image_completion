# -*- coding: utf-8 -*-

"""Updater for Globally Consistent Image Completion"""

import collections
import os
import time
import six

import numpy as np
from numpy import random

import chainer
import chainer.functions as F
from chainer import Variable
from chainer import training
from chainer import reporter as reporter_module
from chainer import serializer as serializer_module
from chainer.training import extension as extension_module
from chainer.training import trigger as trigger_module

import gl_functions as GL_F

class GL_Updater(training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.com, self.dis = kwargs.pop('models')
        self.seed = kwargs.pop('seed')
        self.size_min, self.size_max, self.cut_size = kwargs.pop('resize_param')
        self.mask_min, self.mask_max = kwargs.pop('mask_param')
        self.local_size = kwargs.pop('local_size')
        random.seed(self.seed)
        super(GL_Updater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, completed_y, real_y):
        batchsize = len(completed_y)
        L1 = F.sum(F.softplus(-real_y)) / batchsize
        L2 = F.sum(F.softplus(completed_y)) / batchsize
        loss = L1 + L2 # GAN loss
        chainer.report({'loss': loss}, dis)
        return loss

    # alphaは論文に記載されていた値を使用
    def loss_com(self, com, replaced_x, input_x, completed_y, alpha=0.0004):
        batchsize = len(completed_y)
        MSE     = F.mean_squared_error(replaced_x, input_x)
        fromdis = F.sum(F.softplus(-completed_y)) / batchsize 
        loss = MSE + alpha * fromdis
        chainer.report({'loss': loss}, com)
        return loss
        
    def update(self):

        # optimizer
        com_optimizer = self.get_optimizer('com')
        dis_optimizer = self.get_optimizer('dis')

        # ミニバッチをロード
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        mean_color = self.get_iterator('main').dataset.calc_mean_color()

        # バッチをリサイズ＆カットし，マスク付与
        cut_batch = GL_F.resize_and_cut(batch, size_min=self.size_min,
                                        size_max=self.size_max, cut_size=self.cut_size)
        masked_batch, mask_info = GL_F.make_mask(cut_batch, size_min=self.mask_min, size_max=self.mask_max,
                                                 paint=True, color=mean_color, stack=True)
        
        # input_x:マスク前のカットされた画像 masked_x:マスクされ，マスク領域を示す4チャンネル目が追加された画像
        input_x = Variable(self.converter(cut_batch, self.device))
        masked_x = Variable(self.converter(masked_batch, self.device))

        # 補完・識別ネットワーク
        com, dis = self.com, self.dis

        # 補完ネットワークに入力
        completed_x = com(masked_x)

        # 補完器の出力の補完対象部分以外を入力で置き換える
        replaced_x = GL_F.restore_input(input_x, completed_x, mask_info)
        
        # Local Discriminator に入力する部分を切り出す
        local_replaced_x = GL_F.cut_local(replaced_x, cut_size=self.local_size, mask_info=mask_info)
        local_real_x = GL_F.cut_local(input_x, cut_size=self.local_size)

        # Discriminatorに入力
        completed_y = dis(local_replaced_x, replaced_x)
        real_y      = dis(local_real_x, input_x)
        
        # update実行
        com_optimizer.update(self.loss_com, com, replaced_x, input_x, completed_y)
        dis_optimizer.update(self.loss_dis, dis, completed_y, real_y)

        # イテレーション
        self.iteration += 1
