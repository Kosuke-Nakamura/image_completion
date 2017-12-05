# -*- coding: utf-8 -*-
"""Make sample images with GL-model"""


from __future__ import print_function
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable
import gl_functions as GL_F
from chainer.dataset import convert

from gl_net import CompletionNet
from gl_net import Discriminator
from gl_dataset import GL_ImageDataset, MY_MNIST

def out_sample_image(dataset, com, rows, cols, seed, dst, name, device,
                     out_input=True, out_masked=True, mean_color=None,
                     resize_min=256, resize_max=None, cut_size=256,
                     mask_min=96, mask_max=128):

    np.random.seed(seed)
    n_images = rows * cols
    xp = com.xp
    device = device
    dataset = dataset
    converter = convert.concat_examples
    
    i = np.random.randint(0, len(dataset) - n_images)


    batch = dataset[i:i+n_images]
    if mean_color == None:
        mean_color = dataset.calc_mean_color()

    start_1 = time.time()
    cut_batch = GL_F.resize_and_cut(batch, size_min=resize_min,
                                    size_max=resize_max, cut_size=cut_size)
    masked_batch, mask_info = GL_F.make_mask(cut_batch, size_min=mask_min, size_max=mask_max,
                                             paint=True, color=mean_color, stack=True)
    # デバイス (CPU or GPU)にあわせてvariableを作る
    input_x = Variable(converter(cut_batch, device))
    masked_x = Variable(converter(masked_batch, device))

    start_2 = time.time()
    
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
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3))

    end = time.time()

    print("--Calculation time--")
    print("time1:{} [sec]".format(end - start_1))
    print("time2:{} [sec]".format(end - start_2))
    
    # 保存
    if dst=='': dst='.'
    preview_dir = '{}'.format(dst)
    x_path = preview_dir +\
             '/completed_image_{}.png'.format(name)

    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
        
    Image.fromarray(x).save(x_path)
    print ('save completed image: {}'.format(x_path))

    # 補完前の画像を出力
    if out_input:
        org = np.asarray(np.clip(org * 255, 0.0, 255.0), dtype=np.uint8)
        org = org.reshape((rows, cols, 3, H, W))
        org = org.transpose(0, 3, 1, 4, 2)
        org = org.reshape((rows * H, cols * W, 3))
        org_path = preview_dir +\
                   '/input_image_{}.png'.format(name)
        Image.fromarray(org).save(org_path)
        print ('save original image: {}'.format(org_path))
        
    # マスクを付与した画像を出力
    if out_masked:
        z = np.asarray(np.clip(z * 255, 0.0, 255.0), dtype=np.uint8)
        z = z.reshape((rows, cols, 3, H, W))
        z = z.transpose(0, 3, 1, 4, 2)
        z = z.reshape((rows * H, cols * W, 3))
        z_path = preview_dir +\
                 '/masked_image_{}.png'.format(name)
        Image.fromarray(z).save(z_path)
        print ('save masked image: {}'.format(z_path))
        
def main():
    
    parser = argparse.ArgumentParser(description='Make sample images with GL-model')
    parser.add_argument('--completion_net', '-c', default='',
                        help='Completion network. [.npz] (must)')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files. (must)')
    parser.add_argument('--name', '-n', default='',
                        help='Name of output image')
    parser.add_argument('--out', '-o', default='samples',
                        help='Directory to output the result')
    parser.add_argument('--use_mnist', action='store_true', default=False,
                        help='Use MNIST as train dataset')
    parser.add_argument('--mnist_num', type=int, default=50000,
                        help='Number of MNIST images used as train data')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of images at visualization stage')
    parser.add_argument('--rows', type=int, default=3,
                        help='Number of rows of output image')
    parser.add_argument('--cols', type=int, default=4,
                        help='Number of columns of output image')
    parser.add_argument('--output_original', action='store_true', default=False,
                        help='Output original images if this flag is set')
    parser.add_argument('--output_masked', action='store_true', default=False,
                        help='Output masked images if this flag is set')
    parser.add_argument('--resize_min', type=int, default=256)
    parser.add_argument('--resize_max', type=int, default=None)
    parser.add_argument('--cut_size', type=int, default=256)
    parser.add_argument('--mask_min', type=int, default=96)
    parser.add_argument('--mask_max', type=int, default=128)
    args = parser.parse_args()
    
    print('GPU: {}'.format(args.gpu))
    print('dataset: {}'.format(args.dataset))
    print('rows * columns: {0} * {1}'.format(args.rows, args.cols))
    print('')
    
    # Set up a neural network
    com = CompletionNet()
    chainer.serializers.load_npz(args.completion_net, com)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        com.to_gpu()
        
    # データセットのロード
    if args.use_mnist:
        dataset = MY_MNIST(n=args.mnist_num)
        
    elif args.dataset == '':
        print ("Error:dataset must be specified")
        sys.exit()
        
    else:
        all_files = os.listdir(args.dataset)
        image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]
        print('{} contains {} image files'
              .format(args.dataset, len(image_files)))
        dataset = GL_ImageDataset(paths=image_files, root=args.dataset)
        
    # mean_colorに，学習に使ったモデルの値を使用
    # celeba 1-10000
    # mc = (0.50656409282982351, 0.42607657784298064, 0.38357798138651999)
    # celeba 20001 - 60000
    mc = (0.50562272033542399, 0.42500519392723218, 0.38242747507942842)
    # mc=None
    
    # サンプル画像の作成
    out_sample_image(dataset, com, args.rows, args.cols, args.seed, args.out, args.name, args.gpu,
                     out_input=args.output_original, out_masked=args.output_masked, mean_color=mc,
                     resize_max=args.resize_max, resize_min=args.resize_min,
                     cut_size=args.cut_size, mask_min=args.mask_min, mask_max=args.mask_max)

    
    
if __name__ == '__main__':
    main()
