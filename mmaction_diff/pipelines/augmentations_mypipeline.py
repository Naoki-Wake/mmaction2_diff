import random
from collections.abc import Sequence

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from mmaction.datasets.builder import PIPELINES
from PIL import Image
import os
import math

import numbers
import random
import sys

import cv2
import numpy as np
import skimage
import skimage.transform


INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def rotate(img, angle, resample='BILINEAR', expand=False, center=None):
    imgtype = img.dtype
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    h, w, _ = img.shape
    point = center or (w/2, h/2)
    M = cv2.getRotationMatrix2D(point, angle=-angle, scale=1)

    if expand:
        if center is None:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - point[0]
            M[1, 2] += (nH / 2) - point[1]

            # perform the actual rotation and return the image
            dst = cv2.warpAffine(img, M, (nW, nH))
        else:
            xx = []
            yy = []
            for point in (np.array([0, 0, 1]), np.array([w-1, 0, 1]), np.array([w-1, h-1, 1]), np.array([0, h-1, 1])):
                target = np.matmul(M, point)
                xx.append(target[0])
                yy.append(target[1])
            nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
            nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nw - w)/2
            M[1, 2] += (nh - h)/2
            dst = cv2.warpAffine(img, M, (nw, nh), flags=INTER_MODE[resample])
    else:
        dst = cv2.warpAffine(img, M, (w, h), flags=INTER_MODE[resample])
    return dst.astype(imgtype)


def adjust_brightness(img, brightness_factor):
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.float32) * brightness_factor
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_contrast(img, contrast_factor):
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    im = img.astype(np.float32)
    mean = round(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean())
    im = (1-contrast_factor)*mean + contrast_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_saturation(img, saturation_factor):
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.float32)
    degenerate = cv2.cvtColor(cv2.cvtColor(
        im,
        cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    im = (1-saturation_factor) * degenerate + saturation_factor * im
    im = im.clip(min=0, max=255)
    return im.astype(img.dtype)


def adjust_hue(img, hue_factor):
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    im = img.astype(np.uint8)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
    hsv[..., 0] += np.uint8(hue_factor * 255)

    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB_FULL)
    return im.astype(img.dtype)


def flip(img, y_flip=False, x_flip=False, copy=False):
    assert img.ndim == 3, 'The dimension of image must be 3'
    if y_flip:
        img = img[::-1, :, :]
    if x_flip:
        img = img[:, ::-1, :]
    if copy:
        img = img.copy()
    return img

def random_rotate_img(img, degrees):
    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError('If degrees is a single number,'
                             'must be positive')
        degrees = (-degrees, degrees)
    else:
        if len(degrees) != 2:
            raise ValueError('If degrees is a sequence,'
                             'it must be of len 2.')
    angle = random.uniform(degrees[0], degrees[1])
    return skimage.transform.rotate(img, angle)


def random_rotate_imgs(imgs,
                       degrees):
    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError('If degrees is a single number,'
                             'must be positive')
        degrees = (-degrees, degrees)
    else:
        if len(degrees) != 2:
            raise ValueError('If degrees is a sequence,'
                             'it must be of len 2.')
    angle = random.uniform(degrees[0], degrees[1])
    return [skimage.transform.rotate(img, angle) for img in imgs]


def salt_and_pepper(img, prob=0.01):
    imgtype = img.dtype
    rnd = np.random.rand(img.shape[0], img.shape[1])
    noisy = img.copy()
    noisy[rnd < prob/2] = 0.0
    noisy[rnd > 1 - prob/2] = 255.0
    return noisy.astype(imgtype)


class RandomAugmentation(object):

    def __init__(self, degrees,
                 brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3,
                 p=0.5, prob=0.01):
        self.x_flip = random.choice([True, False])

        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,')
        self.angle = random.uniform(degrees[0], degrees[1])

        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(lambda img: adjust_brightness(img, brightness_factor))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(lambda img: adjust_contrast(img, contrast_factor))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(lambda img: adjust_saturation(img, saturation_factor))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(lambda img: adjust_hue(img, hue_factor))

        self.transforms = transforms

        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        assert isinstance(prob, numbers.Number) and prob >= 0, 'p should be a positive value'
        self.p = p
        self.prob = prob

    def __call__(self, img):
        img = img.copy()
        for trans in self.transforms:
            img = trans(img)

        if random.random() < self.p:
            img = salt_and_pepper(img, self.prob)

        # img = flip(img, y_flip=False, x_flip=self.x_flip, copy=False)
        img = rotate(img, self.angle)
        img = np.array(np.clip(img, 0, 255),
                 dtype=np.uint8)
        return img

@PIPELINES.register_module()
class VideoAug:
    def __init__(self, degrees = 10, prob = 0.01):
        self.degrees = degrees
        self.prob = prob
        #self.count = 10
        pass
    def __call__(self, results):
        #import pdb
        #pdb.set_trace()
        """print(results.keys())
        print(type(results))
        print(len(results['imgs']))
        print(type(results['imgs'][0]))
        print(np.shape(results['imgs'][0]))
        print(results['filename_tmpl'])
        print(results['label'])
        print(results['frame_dir'])
        print("/mmaction2/data/demo/" + os.path.basename(results['frame_dir']) + ".jpg")"""
        #if 'aug' not in results['frame_dir']:
            #print('hoge')
        #ra = RandomAugmentation(30)
        ra = RandomAugmentation(self.degrees, prob = self.prob)
        for i, frame in enumerate(results['imgs']):
            results['imgs'][i] = ra(frame)
            #im = Image.fromarray(results['imgs'][0])
            #im.save("/mmaction2/data/demo/" + os.path.basename(results['frame_dir']) + ".jpg")

        """if self.count < 1:
            frame_rate = 30.0 # フレームレート
            size = (640, 360) # 動画の画面サイズ
            fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
            writer = cv2.VideoWriter("/mmaction2/data/demo/" + os.path.basename(results['frame_dir']) + "_aug.mp4", fmt, frame_rate, size) # ライター作成
            for frame in results['imgs']:
                #frame = ra(frame)
                writer.write(frame) # 画像を1フレーム分として書き込み
            writer.release() # ファイルを閉じる
        self.count = self.count + 1"""
        return results

@PIPELINES.register_module()
class MyDebug:
    def __init__(self, note = ''):
        self.count = 0
        self.note = note
    def __call__(self, results):
        if self.count < 5:
            #print(len(results['imgs']))
            print(np.shape(results['imgs'][0]))
            print(np.shape(results['imgs']))
            '''frame_rate = 30.0 # フレームレート
            size = (224, 224) # 動画の画面サイズ
            fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4)
            writer = cv2.VideoWriter("/mmaction2/data/demo/20210521/" + os.path.basename(results['frame_dir']) + "_resize.mp4", fmt, frame_rate, size) # ライター作成
            for frame in results['imgs']:
                writer.write(frame) # 画像を1フレーム分として書き込み
            writer.release() # ファイルを閉じる'''
            print(results['imgs'][0][1][1][1])
            cv2.imwrite("/mmaction2/data/demo/20210522/" + os.path.basename(results['frame_dir']) + "_" + self.note + ".jpg", results['imgs'][0])
        self.count = self.count + 1
        print('debug:'+str(self.count))
        return results