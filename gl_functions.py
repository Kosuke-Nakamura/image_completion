# -*- coding: utf-8 -*-

"""Functions for Globally and Locally Consistent Image Completion"""

import os
import time
from scipy import misc

import numpy
from numpy import random

import chainer
from chainer import Variable
from chainer import function
from chainer import utils
from chainer import cuda


# リサイズして切り取る
def resize_and_cut(x, size_min=256, size_max=None, cut_size=256):
    """Resize and cut input images
    
    入力した画像(array)を，一定範囲でランダムにリサイズし，
    一定サイズでカットする
    リサイズは画像の短辺の長さを一定の範囲内でランダムに変更し，
    縦横比が変わらないようにもう１辺の長さに変更する
    
    size_min: リサイズ後の短辺の長さの値の範囲の下限
              default: 256
    size_max: リサイズ後の短辺の長さの値の範囲の上限
              default: size_min * 1.5
    cut_size: cut_size * cut_size の正方領域をカットする
              size_min >= cut_size
              defult: 256"""
    
    
    batchsize = len(x)
    out = []

    # 引数チェック，設定
    if not size_max:
        size_max = int(1.5 * size_min)
    else:
        assert size_max >= size_min, "size_max must be larger or equal than size_min"
    
    if cut_size > size_min:
        cut_size = size_min
        
    for i in range(0, batchsize):
        height, width = x[i][0].shape # Rチャンネルの行数，列数
        
        # 短い方の辺の長さを[size_min, size_max]の範囲にランダムに変更
        if  width > height:
            new_h = random.randint(size_min, size_max+1)
            new_w = int(width * new_h / height) # 縦横比はそのまま
        else:
            new_w = random.randint(size_min, size_max+1)
            new_h = int(height * new_w / width)

        new_size = (new_h, new_w)

        # 各チャンネルをリサイズ
        resized_R = misc.imresize(x[i][0], new_size)
        resized_G = misc.imresize(x[i][1], new_size)
        resized_B = misc.imresize(x[i][2], new_size)

        # 切り取る位置（切り取る領域の左上）
        cut_h = random.randint(0, new_h - cut_size + 1)
        cut_w = random.randint(0, new_w - cut_size + 1)
        
        cut_image = [resized_R[cut_h:cut_h+cut_size, cut_w:cut_w+cut_size],
                     resized_G[cut_h:cut_h+cut_size, cut_w:cut_w+cut_size],
                     resized_B[cut_h:cut_h+cut_size, cut_w:cut_w+cut_size]]

        out.append(cut_image)

    out = numpy.array(out, dtype=numpy.float32) / 255.

    return out
        
# バッチサイズ分だけマスクを生成
# 現在のバッチの画像のサイズを元にランダムにマスクを生成
# paintがTrueなら入力画像の対応部分を塗りつぶす
# stackがTrueなら入力画像の4チャンネル目としてマスクを追加したものを返す
# x[ndarray], paint[boolean], color(float, float, float), stack[boolean]
def make_mask(x, size_min=96, size_max=128, paint=True, color=(0., 0., 0.), stack=True):


    assert size_max >= size_min, "size_max must be larger or equal than size_min"
    
    
    batchsize = len(x)
    masks = []
    mask_info = {'index':[], 'size':[]}

    for i in range(0, batchsize):
        height, width = x[i][0].shape
        mask = numpy.zeros((height, width), dtype=numpy.float32)

        # マスクのサイズをランダムに決定
        mask_h = random.randint(size_min, size_max+1)
        mask_w = random.randint(size_min, size_max+1)

        # マスクの位置(領域の左上)をランダムに決定
        mask_ind_h = random.randint(0, height - mask_h + 1)
        mask_ind_w = random.randint(0, width - mask_w + 1)

        # マスク配列のマスク部分を1で埋める
        mask[mask_ind_h:mask_ind_h + mask_h, mask_ind_w:mask_ind_w + mask_w] = 1.
        
        masks.append(mask)
        
        mask_info['index'].append((mask_ind_h, mask_ind_w))
        mask_info['size'].append((mask_h, mask_w))
        
    # paintがTrueであれば入力画像のマスクに対応する部分を塗りつぶす
    if paint:
        x_painted = numpy.copy(x)

        for i in range(0, batchsize):
            tmp_slice = (slice(mask_info['index'][i][0], mask_info['index'][i][0]+mask_info['size'][i][0]),
                         slice(mask_info['index'][i][1], mask_info['index'][i][1]+mask_info['size'][i][1]))
            x_painted[i][0][tmp_slice] = color[0]
            x_painted[i][1][tmp_slice] = color[1]
            x_painted[i][2][tmp_slice] = color[2]       

    # stackがTrueであれば4チャンネル目としてマスクを追加して返す
    if stack:
        if paint:
            x_stacked = x_painted.tolist()
        else:
            x_stacked = x.tolist()
        
        for i in range(0, batchsize):
            # maskをチャンネル数1の3次元配列に (numpy.append用)
            mask.reshape(1, height, width)
            # i+1番目の画像の4チャンネル目としてマスクを追加
            x_stacked[i].append(masks[i])

        x_stacked = numpy.array(x_stacked, dtype=numpy.float32)

        return x_stacked, mask_info

    elif paint:
        return x_painted, mask_info
    
    else:
        masks = numpy.array(masks, dtype=numpy.float32)
        return masks, mask_info
  

# 補完されたミニバッチを受け取って局所識別器に入力する部分を切り取る
# mask_infoが入力された場合はマスクされた領域の中心を中心とするような128 * 128の領域を切り取る
# mask_infoが入力されない場合はランダムで128 * 128に切り取る
# 引数がVariableなのでfunction.Functionを継承して作っている
class CutLocal(function.Function):

    def __init__(self, mask_info, cut_size):

        self.mask_info = mask_info
        self.cut_size = cut_size

    def forward_with_mask(self, xs):
        
        # self.retain_inputs(())
        ary = xs[0]
        self.batchsize = len(ary)

        xp = cuda.get_array_module(*xs)
        
        self.slices = []
        self._in_shape = ary.shape
        self._in_dtype = ary.dtype
        out_ary = xp.zeros((self.batchsize, 3, self.cut_size, self.cut_size), dtype=self._in_dtype)
        
        for i in range(0, self.batchsize):

            _, h, w = ary[i].shape
            
            mask_ind_h, mask_ind_w = self.mask_info['index'][i]
            mask_h, mask_w = self.mask_info['size'][i]
            
            # マスクの中心の座標
            mask_c_h = mask_ind_h + mask_h // 2
            mask_c_w = mask_ind_w + mask_w // 2
            
            # (mask_c_h, mask_c_w)に中心が来るような128*128の領域の左上隅の座標
            cut_ind_h = mask_c_h - self.cut_size // 2
            cut_ind_w = mask_c_w - self.cut_size // 2
            # 領域が画像をはみ出していたらずらす
            # 縦
            if cut_ind_h + self.cut_size >= h:
                cut_ind_h = h - self.cut_size
            elif cut_ind_h < 0:
                cut_ind_h = 0
            # 横    
            if cut_ind_w + self.cut_size >= w:
                cut_ind_w = w - self.cut_size
            elif cut_ind_w < 0:
                cut_ind_w = 0

            # 切り取った位置を保存しておく
            self.slices.append((slice(0,3),
                                slice(cut_ind_h, cut_ind_h+self.cut_size),
                                slice(cut_ind_w, cut_ind_w+self.cut_size)))
                
            out_ary[i] = ary[i][self.slices[i]]            
            
        return out_ary,
        
    def forward_random(self, xs):
        
        # self.retain_inputs(())
        ary = xs[0]
        self.batchsize = len(ary)

        xp = cuda.get_array_module(*xs)
        
        self.slices = []
        self._in_shape = ary.shape
        self._in_dtype = ary.dtype
        out_ary = xp.zeros((self.batchsize, 3, self.cut_size, self.cut_size), dtype=self._in_dtype)

        for i in range(0, self.batchsize):
            _, h, w = ary[i].shape

            # 切り取る領域の左上の点をランダムに決定
            cut_ind_h = random.randint(0, h - self.cut_size + 1)
            cut_ind_w = random.randint(0, w - self.cut_size + 1)

            self.slices.append((slice(0,3),
                                slice(cut_ind_h, cut_ind_h + self.cut_size),
                                slice(cut_ind_w, cut_ind_w + self.cut_size)))
                
            out_ary[i] = ary[i][self.slices[i]]
           
        return out_ary,
        

    def forward(self, xs):

        if self.mask_info:
            return self.forward_with_mask(xs)
        else:
            return self.forward_random(xs)
        
    def backward(self, xs, gys):
        xp = cuda.get_array_module(*gys)
        gy = gys[0]
        gx = xp.zeros(self._in_shape, self._in_dtype)
        
        for i in range(0, self.batchsize):
            gx[i][self.slices[i]] = gy[i]

        return gx,
                

def cut_local(x, cut_size=128, mask_info=None):
    """Extract elements around masked region or randomly

    mask_infoがNoneのときは入力画像からランダムでcut_size * cut_sizeの領域を切り取る
    mask_infoがあるときは，マスクされた領域の中心を中心とする128 * 128の領域を切り取る"""

    return CutLocal(mask_info, cut_size)(x)
    

class RestoreInput(function.Function):

    def __init__(self, mask_info, x_org):
        self.mask_info = mask_info
        self.x_org     = x_org.data
        self.batchsize = len(mask_info['index'])
        
    def forward(self, xs):

        # self.retain_inputs(())

        completed = xs[0]
        original  = self.x_org
        out_ary = cuda.copy(original)
        
        self.slices = []
        self._in_shape = original.shape
        self._in_dtype = original.dtype
        
        for i in range(0, self.batchsize):
            mask_ind_h, mask_ind_w = self.mask_info['index'][i]
            mask_h, mask_w = self.mask_info['size'][i]
            
            # マスクされた部分をスライス表記して保存しておく
            self.slices.append((slice(0, 3),
                                slice(mask_ind_h, mask_ind_h + mask_h),
                                slice(mask_ind_w, mask_ind_w + mask_w)))

            # 元画像のマスクされる領域のみを補完器の出力に置き換える
            # <--> 補完器の出力のマスクされた領域以外を元の画像で置きかえる
            out_ary[i][self.slices[i]] = completed[i][self.slices[i]]

        
        return out_ary,

    def backward(self, xs, gys):
        xp = cuda.get_array_module(*gys)
        gy = gys[0]
        gx = []

        # 出力する勾配を初期化
        gx = xp.zeros(self._in_shape, self._in_dtype) ##1
        
        # 補完された領域以外は入力画像に置き換えてあるから
        # 補完された部分に対応する勾配だけ伝達する
        for i in range(0, self.batchsize):
            gx[i][self.slices[i]] = gy[i][self.slices[i]]

        return gx,

def restore_input(x_org, x_com, mask_info):
    """Restore input image

    補完器の出力x_comのうちmask_infoに書かれた補完の対象領域以外を元の入力画像x_orgの値に置き換える"""

    return RestoreInput(mask_info, x_org)(x_com)
