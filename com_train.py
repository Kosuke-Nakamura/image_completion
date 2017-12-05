# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import sys

import chainer
from chainer import training
from chainer.training import extensions

from gl_visualize import out_generated_image

from gl_net import CompletionNet
from com_updater import Com_Updater
from gl_dataset import GL_ImageDataset, MY_MNIST


def main():
    parser = argparse.ArgumentParser(description='Globally and Locally Consitent Image Completion')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files. (must)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of images at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--use_mnist', action='store_true', default=False,
                        help='Use MNIST as train dataset')
    parser.add_argument('--mnist_num', type=int, default=50000,
                        help='Number of MNIST images used as train data')
    parser.add_argument('--resize_min', type=int, default=256)
    parser.add_argument('--resize_max', type=int, default=None)
    parser.add_argument('--cut_size', type=int, default=256)
    parser.add_argument('--mask_min', type=int, default=96)
    parser.add_argument('--mask_max', type=int, default=128)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    com = CompletionNet()

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        com.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    def make_optimizer(model, rho=0.95): # rhoはとりあえずDCGANと同じ値に
        optimizer = chainer.optimizers.AdaDelta(rho=rho)
        optimizer.setup(model)
        # DCGANサンプルにはあったがとりあえず除く
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    opt_com = make_optimizer(com)

    # データセットのロード
    if args.use_mnist:
        train = MY_MNIST(n=args.mnist_num)
        
    elif args.dataset == '':
        print ("Error:dataset must be specified")
        sys.exit()
        
    else:
        all_files = os.listdir(args.dataset)
        image_files = [f for f in all_files if ('png' in f or 'jpg' in f)]
        print('{} contains {} image files'
              .format(args.dataset, len(image_files)))
        train = GL_ImageDataset(paths=image_files, root=args.dataset)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Set up a trainer
    updater = Com_Updater(
        models=(com),
        iterator=train_iter,
        optimizer={
            'com': opt_com},
        device=args.gpu,
        seed=100,
        resize_param=(args.resize_min, args.resize_max, args.cut_size),
        mask_param=(args.mask_min, args.mask_max))
    print('Updater is set up')
    
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        com, 'com_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'com/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            com,
            4, 3, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    print ('Running the Training...')
    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
