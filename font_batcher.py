"""
taken and modified from https://github.com/pranv/ARC
"""

import os
import numpy as np
from numpy.random import choice
import torch
from torch.autograd import Variable

from scipy.misc import imresize as resize

from image_augmenter import ImageAugmenter

use_cuda = False


class Fonts(object):
    def __init__(self, path=os.path.join('font_imgs', 'font_imgs.npy'), batch_size=64, image_size=32):
        """
        batch_size: the output is (2 * batch size, 1, image_size, image_size)
                    X[i] & X[i + batch_size] are the pair
        image_size: size of the image
        data_split: in number of alphabets, e.g. [30, 10] means out of 50 Omniglot characters,
                    30 is for training, 10 for validation and the remaining(10) for testing
        within_alphabet: for verfication task, when 2 characters are sampled to form a pair,
                        this flag specifies if should they be from the same alphabet/language
        ---------------------
        Data Augmentation Parameters:
            flip: here flipping both the images in a pair
            scale: x would scale image by + or - x%
            rotation_deg
            shear_deg
            translation_px: in both x and y directions
        """

        num_chars_in_font = 62  # num of chars used
        num_fonts = 65  # num fonts
        num_chars_instances = 65*62  # 4030 num char instances, which is num_chars * num_fonts
        num_samples_per_char = 100  # samples per char

        chars = np.load(path)

        # resize the images
        resized_chars = np.zeros((num_chars_instances, num_samples_per_char, image_size, image_size), dtype='uint8')
        for i in range(num_chars_instances):
            for j in range(num_samples_per_char):
                resized_chars[i, j] = resize(chars[i, j], (image_size, image_size))
        chars = resized_chars

        self.mean_pixel = chars.mean() / 255.0  # used later for mean subtraction

        # start of each alphbt in a list of chars
        a_start = []
        for i in range(num_fonts):
            a_start.append(i * num_chars_in_font)

        a_size = [62] * num_fonts

        # each alphabet/language has different number of characters.
        # in order to uniformly sample all characters, we need weigh the probability
        # of sampling a alphabet by its size. p is that probability
        def size2p(size):
            s = np.array(size).astype('float64')
            return s / s.sum()

        self.size2p = size2p

        self.num_samples_per_char = num_samples_per_char
        self.num_chars_in_font = num_chars_in_font
        self.num_fonts = num_fonts
        self.data = chars
        self.a_start = a_start
        self.a_size = a_size
        self.image_size = image_size
        self.batch_size = batch_size

    def fetch_batch(self, part):
        """
            This outputs batch_size number of pairs
            Thus the actual number of images outputted is 2 * batch_size
            Say A & B form the half of a pair
            The Batch is divided into 4 parts:
                (Dissimilar A, Dissimilar B)
                (Similar A, Similar B)

            Corresponding images in Similar A and Similar B form the similar pair
            similarly, Dissimilar A and Dissimilar B form the dissimilar pair

            When flattened, the batch has 4 parts with indices:
                Dissimilar A 		0 - batch_size / 2
                Similar A    		batch_size / 2  - batch_size
                Dissimilar B 		batch_size  - 3 * batch_size / 2
                Similar B 			3 * batch_size / 2 - batch_size

        """
        pass


class Batcher(Fonts):
    def __init__(self, path=os.path.join('font_imgs', 'font_imgs.npy'), batch_size=64, image_size=32):
        Fonts.__init__(self, path, batch_size, image_size)

        a_start = self.a_start
        a_size = self.a_size

        # slicing indices for splitting a_start & a_size
        i = 40  # val start
        j = 50  # test start
        starts = {}
        starts['train'], starts['val'], starts['test'] = a_start[:i], a_start[i:j], a_start[j:]
        
        sizes = {}
        sizes['train'], sizes['val'], sizes['test'] = a_size[:i], a_size[i:j], a_size[j:]

        size2p = self.size2p

        p = {}
        p['train'], p['val'], p['test'] = size2p(sizes['train']), size2p(sizes['val']), size2p(sizes['test'])

        self.starts = starts
        self.sizes = sizes
        self.p = p

    def fetch_batch(self, part, batch_size: int = None):

        if batch_size is None:
            batch_size = self.batch_size

        X, Y = self._fetch_batch(part, batch_size)  # array of single images

        # print('X shape before tensor conversion', X.shape)

        X = Variable(torch.from_numpy(X)).view(2*batch_size, self.image_size, self.image_size)

        # print('X shape after reshape', X.shape)

        X1 = X[:batch_size]  # (B, h, w)
        X2 = X[batch_size:]  # (B, h, w)

        X = torch.stack([X1, X2], dim=1)  # (B, 2, h, w)  # array of image PAIRS now!!

        # print('X shape after stack', X.shape)

        Y = Variable(torch.from_numpy(Y))

        if use_cuda:
            X, Y = X.cuda(), Y.cuda()

        return X, Y

    def _fetch_batch(self, part, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        data = self.data
        starts = self.starts[part]  # provides font starts for each part of dataset (train/val/test)
        sizes = self.sizes[part]  # sizes tells you the number of fonts in each dataset part
        p = self.p[part]  # probability of sampling a font
        image_size = self.image_size

        num_fonts = len(starts)  # number of alphabets

        # fill up this matrix with the batch (pairs of images, size, size)
        X = np.zeros((2 * batch_size, image_size, image_size), dtype='uint8')

        # loop through half the batch size, since we'll fill up 2 pairs at a time
        # 1 similar pair, and 1 dissimilar pair
        for i in range(batch_size // 2):

            # choose similar chars
            # choose similar fonts

            # choose similar chars
            # choose different fonts

            # choose similar chars.  choose char idx from start to end of font idxs
            same_char_idx = choice(self.num_chars_in_font)  # this is the offset from the font start
            same_font_idx = choice(starts)  # choose a font for the similar case
            same_sample_idx1, same_sample_idx2 = choice(self.num_samples_per_char, 2)  # choose 2 diff samples from the char

            # choose similar chars.
            diff_char_idx = choice(self.num_chars_in_font)  # this is the offset from the font starts
            diff_font_idx1, diff_font_idx2 = choice(starts, 2)
            diff_sample_idx1, diff_sample_idx2 = choice(self.num_samples_per_char, 2)

            # similar font pair, note:  the char offset is the same though, and sample num is diff
            X[i + batch_size // 2] = data[same_font_idx + same_char_idx, same_sample_idx1]
            X[i + 3 * batch_size // 2] = data[same_font_idx + same_char_idx, same_sample_idx2]

            # dissimilar font pair, note:  the char offset is the same though, and sample num is diff
            X[i] = data[diff_font_idx1 + diff_char_idx, diff_sample_idx1]
            X[i + batch_size] = data[diff_font_idx2 + diff_char_idx, diff_sample_idx2]


        y = np.zeros((batch_size, 1), dtype='int32')
        y[:batch_size // 2] = 0  # first half are diff imgs
        y[batch_size // 2:] = 1  # second half are same imgs

        X = X / 255.0

        X = X - self.mean_pixel

        # print('X before new axis in _fetch batch', X.shape)

        X = X[:, np.newaxis]
        X = X.astype("float32")

        # print('X shape in _fetch batch', X.shape)

        return X, y
