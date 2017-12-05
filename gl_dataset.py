# -*- coding: utf-8 -*-

import os
import gc

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import six

import chainer
from chainer.dataset import dataset_mixin


def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = numpy.asarray(f, dtype=dtype)
    except TypeError:
        print ("TypeError!!!")
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image

# 画像を読み込んでfloat32の配列として格納
class GL_ImageDataset(dataset_mixin.DatasetMixin):

    """画像ファイルへのパスからなる画像データセット

    画像のパスのリストを保持し，__getitem__によりインデックスに対応したパスの画像をNumPy配列として
    読み込んで返します．
    chainer.datasets.ImageDatasetsに以下の機能を追加した
    
    正規化
        読み込み時に返す画像の値を，引数scaleに対し，[0, max * scale/255]の範囲に正規化
        scaleのデフォルト値はscale=1.0 (元データの255を1.0に対応させる)

    calc_mean_color
        データセット全体の各チャンネルのピクセル値の平均値を計算し，
        (Rの平均値，Gの平均値，Bの平均値)
        の形で値を返す

    This dataset reads an external image file on every call of the
    :meth:`__getitem__` operator. The paths to the image to retrieve is given
    as either a list of strings or a text file that  contains paths in distinct
    lines.

    Each image is automatically converted to arrays of shape
    ``channels, height, width``, where ``channels`` represents the number of
    channels in each pixel (e.g., 1 for grey-scale images, and 3 for RGB-color
    images).

    .. note::
       **This dataset requires the Pillow package being installed.** In order
       to use this dataset, install Pillow (e.g. by using the command ``pip
       install Pillow``). Be careful to prepare appropriate libraries for image
       formats you want to use (e.g. libpng for PNG images, and libjpeg for JPG
       images).

    Args:
        paths (str or list of strs): If it is a string, it is a path to a text
            file that contains paths to images in distinct lines. If it is a
            list of paths, the ``i``-th element represents the path to the
            ``i``-th image. In both cases, each path is a relative one from the
            root path given by another argument.
        root (str): Root directory to retrieve images from.
        dtype: Data type of resulting image arrays.

    """

    def __init__(self, paths, root='.', scale=1., dtype=numpy.float32):
        _check_pillow_availability()
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._dtype = dtype
        self._scale = scale
        self._mean_calculated = False
        # デバッグ用 （計算に時間がかかるため予め計算した値を使う）
        # self._mean_calculated = True
        
    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        image = _read_image_as_array(path, self._dtype)

        # 正規化
        image *= self._scale / 255.
        
        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, numpy.newaxis]
            
        return image.transpose(2, 0, 1)


    def calc_mean_color(self):

        # 初めてこれを呼び出したときに平均値を計算
        if not self._mean_calculated:
            # RGB各チャンネルの平均値
            meanR, meanG, meanB = 0., 0., 0.
            length = len(self._paths)
            print 'length = {}'.format(length)

            # 画像ごとの平均値を加算していく
            print 'calculating mean colors of dataset...'
            for i in range(0, length):
                if i % 1000 == 0:
                    print '{}% done'.format(1.0 * i / length * 100)  
                image = self.get_example(i)
                meanR += numpy.mean(image[0])
                meanG += numpy.mean(image[1])
                meanB += numpy.mean(image[2])

            # 全体の平均値
            self.mean_colors = (meanR / length,
                                meanG / length,
                                meanB / length)
            print 'mean colors = {}'.format(self.mean_colors)
            
            # 平均値計算済み
            self._mean_calculated = True

        # デバッグ用 celebAの全ファイルに対して予め計算した結果
        # self.mean_colors = (0.50612347387592882, 0.42543392151575721, 0.38283131619313976)
            
        return self.mean_colors
                

def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))


class MY_MNIST(dataset_mixin.DatasetMixin):
    """ Get MNIST train data for GL-model

    chainer.datasets.get_mnistを用いてMNISTの学習データをロードする
    画像は各チャンネルに同じ値をもつ3チャンネルの画像として読み込まれる

    n: 何枚使うかを指定 (default: 50000)

    """

    def __init__(self, n=50000):

        # MNISTのtrainデータを読み込み
        tmp, _  = chainer.datasets.get_mnist(withlabel=False, ndim=2, scale=1.0)
        self.data = numpy.zeros((n, 28, 28), dtype=numpy.float32)
        self.data = tmp[0:n]

        del tmp, _
        gc.collect()
        
        self._mean_calculated = False
        
    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        image = numpy.zeros((3, 28, 28), dtype=numpy.float32)
        image[0] = self.data[i]
        image[1] = self.data[i]
        image[2] = self.data[i]
        
        return image

    def calc_mean_color(self):

        # 初めてこれを呼び出したときに平均値を計算
        if not self._mean_calculated:
            # 平均値 (3チャンネルとも同じ値)
            mean = 0.
            length = len(self.data)
            print 'length = {}'.format(length)

            # 画像ごとの平均値を加算していく
            print 'calculating mean colors of dataset...'
            for i in range(length):
                if i % 1000 == 0:
                    print '{}% done'.format(1.0 * i / length * 100)  
                image = self.data[i]
                mean += numpy.mean(image)
                
            # 全体の平均値
            self.mean_colors = (mean / length,
                                mean / length,
                                mean / length)
            print 'mean colors = {}'.format(self.mean_colors)
            
            # 平均値計算済み
            self._mean_calculated = True

            
        return self.mean_colors
