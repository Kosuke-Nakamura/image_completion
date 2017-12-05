# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
import gl_functions as GL_F

def out_generated_image(com, rows, cols, seed, dst,
                        resize_min=256, resize_max=None, cut_size=256,
                        mask_min=96, mask_max=128):
    
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = com.xp
        converter = trainer.updater.converter
        device = trainer.updater.device
        dataset = trainer.updater.get_iterator('main').dataset
        
        i = np.random.randint(0, len(dataset) - n_images)
        batch = dataset[i:i+n_images]
        mean_color = dataset.calc_mean_color()
        cut_batch = GL_F.resize_and_cut(batch, size_min=resize_min,
                                        size_max=resize_max, cut_size=cut_size)
        masked_batch, mask_info = GL_F.make_mask(cut_batch, size_min=mask_min, size_max=mask_max,
                                                 paint=True, color=mean_color, stack=True)
        # デバイス (CPU or GPU)にあわせてvariableを作る
        input_x = Variable(converter(cut_batch, device))
        masked_x = Variable(converter(masked_batch, device))

        # 補完器に入力
        # with chainer.using_config('train', False): # v2以降の仕様のようだ
        #    completed_x = com(masked_x)
        completed_x = com(masked_x)
        # 補完対象部分以外を元の画像で置き換える
        replaced_x = GL_F.restore_input(input_x, completed_x, mask_info)    
            
        x = chainer.cuda.to_cpu(replaced_x.data)
        z = masked_batch[:, 0:3]
        org = cut_batch
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        z = np.asarray(np.clip(z * 255, 0.0, 255.0), dtype=np.uint8)
        org = np.asarray(np.clip(org * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))
        z = z.reshape((rows, cols, 3, H, W))
        z = z.transpose(0, 3, 1, 4, 2)
        z = z.reshape((rows * H, cols * W, 3))
        org = org.reshape((rows, cols, 3, H, W))
        org = org.transpose(0, 3, 1, 4, 2)
        org = org.reshape((rows * H, cols * W, 3))

        preview_dir = '{}/preview'.format(dst)
        x_path = preview_dir +\
            '/completed_image{:0>8}.png'.format(trainer.updater.iteration)
        z_path = preview_dir +\
            '/masked_image_{:0>8}.png'.format(trainer.updater.iteration)
        org_path = preview_dir +\
            '/input_image_{:0>8}.png'.format(trainer.updater.iteration)

        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(x_path)
        Image.fromarray(z).save(z_path)
        Image.fromarray(org).save(org_path)
    return make_image
