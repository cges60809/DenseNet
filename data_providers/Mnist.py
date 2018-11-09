#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:45:13 2018

@author: islab
"""
import tempfile
import os
import pickle
import random
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from .base_provider import ImagesDataSet, DataProvider
import numpy as np

class selfDataSet(ImagesDataSet):
    def __init__(self, images, labels, n_classes, shuffle, normalization,
                 augmentation):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            n_classes: `int`, number of cifar classes - 10 or 100
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            augmentation: `bool`
        """
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_images_and_labels(images, labels)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization
        self.images = self.normalize_images(images, self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(
                self.images, self.labels)
        else:
            images, labels = self.images, self.labels
       
        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice



class MNISTINDataProvider(DataProvider):
    """Abstract class for cifar readers"""

    def __init__(self, save_path=None, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None,
                 one_hot=True, **kwargs):
        """
        Args:
            save_path: `str`
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return lasels one hot encoded
        """
        self._save_path = save_path
        self.one_hot = one_hot


        # add train and validations and test datasets
        data_dir = 'temp'
        mnist = read_data_sets(data_dir)        
        trainimage = np.array([np.reshape(x, (28,28,1)) for x in mnist.train.images])
        validationimage = np.array([np.reshape(x, (28,28,1)) for x in mnist.validation.images])
        testimage = np.array([np.reshape(x, (28,28,1)) for x in mnist.test.images])
        train_labels = mnist.train.labels
        validation_labels = mnist.validation.labels
        test_labels = mnist.test.labels

        #images, labels = self.read_cifar(train_fnames)
        if validation_set is not None:
            self.train = selfDataSet(
                images=trainimage, labels=self.label_convert(train_labels),
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
            self.validation = selfDataSet(
                images=validationimage, labels=self.label_convert(validation_labels),
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
        else:
            self.train = selfDataSet(
                images=trainimage, labels=self.label_convert(train_labels),
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)

        # add test set
        self.test = selfDataSet(
            images=testimage, labels=self.label_convert(test_labels),
            shuffle=None, n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(
                tempfile.gettempdir(), 'Mnist%d' % self.n_classes)
        return self._save_path

    @property
    def data_shape(self):
        return (28, 28, 1)

    @property
    def n_classes(self):
        return self._n_classes

    def get_filenames(self, save_path):
        """Return two lists of train and test filenames for dataset"""
        raise NotImplementedError

    def label_convert(self, mnistlabel):
        label = np.zeros([mnistlabel.size,10])
        for i in range(mnistlabel.size):
            label[i,mnistlabel[i]] = 1
        return label

class MNISTDataProvider(MNISTINDataProvider):
    _n_classes = 10
    data_augmentation = False

    def get_filenames(self, save_path):
        sub_save_path = os.path.join(save_path, 'cifar-10-batches-py')
        train_filenames = [
            os.path.join(
                sub_save_path,
                'data_batch_%d' % i) for i in range(1, 6)]
        test_filenames = [os.path.join(sub_save_path, 'test_batch')]
        return train_filenames, test_filenames



