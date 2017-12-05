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


class Com_Updater(training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.com = kwargs.pop('models')
        self.seed = kwargs.pop('seed')
        self.size_min, self.size_max, self.cut_size = kwargs.pop('resize_param')
        self.mask_min, self.mask_max = kwargs.pop('mask_param')
        random.seed(self.seed)
        super(Com_Updater, self).__init__(*args, **kwargs)

    def loss_com_MSE(self, com, replaced_x, input_x):
        # 損失を計算
        loss = F.mean_squared_error(replaced_x, input_x)
        # 損失を表示するためにreport
        chainer.report({'loss': loss}, com)
        # 出力
        return loss
        
    def update(self):
        com_optimizer = self.get_optimizer('com')

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        mean_color = self.get_iterator('main').dataset.calc_mean_color()

        cut_batch = GL_F.resize_and_cut(batch, size_min=self.size_min,
                                        size_max=self.size_max, cut_size=self.cut_size)
        masked_batch, mask_info = GL_F.make_mask(cut_batch, size_min=self.mask_min, size_max=self.mask_max,
                                                 paint=True, color=mean_color, stack=True)

        # input_x:マスク前のカットされた画像 masked_x:マスクされ，マスク領域を示す4チャンネル目が追加された画像
        # converterでCPU or GPU 用に変換
        input_x = Variable(self.converter(cut_batch, self.device))
        masked_x = Variable(self.converter(masked_batch, self.device))

        # 補完器に入力
        com = self.com
        completed_x = com(masked_x)
        
        # 出力の補完対象部分以外を入力で置き換える
        replaced_x = GL_F.restore_input(input_x, completed_x, mask_info)

        # update実行
        com_optimizer.update(self.loss_com_MSE, com, replaced_x, input_x)
        
        # イテレーション
        self.iteration += 1
        
