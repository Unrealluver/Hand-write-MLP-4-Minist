#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import math
import struct
import scipy.special
import h5py


class MLP(object):
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
        # initial
        self.w1 = np.random.randn(self.hiddennodes, self.inputnodes) * 0.01
        self.w2 = np.random.randn(self.outputnodes, self.hiddennodes) * 0.01
        self.b1 = np.zeros((200, 1))
        self.b2 = np.zeros((10, 1))
        self.epoch = 1

    def softmax(self, x):
        from scipy.special import expit
        return expit(x)

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def loss_function(self, origin_label, fp_result):
        return -origin_label * (math.log2(fp_result)) - (1 - origin_label) * (math.log2(1 - fp_result))

    def forward_propagation(self, input_data, weight_matrix, b):
        z = np.add(np.dot(weight_matrix, input_data), b)
        return self.softmax(z)

    def back_propagation(self, a, z, da, weight_matrix, b):
        dz = da * (z * (1 - z))
        weight_matrix -= self.learningrate * np.dot(dz, a.T) / 60000
        b -= self.learningrate * np.sum(dz, axis=1, keepdims=True) / 60000
        da_n = np.dot(weight_matrix.T, da)
        return da_n

    def train(self, input_data, label_data):
        for item in range(self.epoch):
            print('iter: %d' % item)
            for i in range(60000):
                # fore
                a1 = self.forward_propagation(input_data[:, i].reshape(-1, 1), self.w1, self.b1)
                a2 = self.forward_propagation(a1, self.w2, self.b2)
                # cal
                dz2 = a2 - label_data[:, i].reshape(-1, 1)
                dz1 = np.dot(self.w2.T, dz2) * a1 * (1.0 - a1)
                # back
                self.w2 -= self.learningrate * np.dot(dz2, a1.T)
                self.b2 -= self.learningrate * dz2

                self.w1 -= self.learningrate * np.dot(dz1, (input_data[:, i].reshape(-1, 1)).T)
                self.b1 -= self.learningrate * dz1

    def predict(self, input_data, label):
        precision = 0
        for i in range(10000):
            a1 = self.forward_propagation(input_data[:, i].reshape(-1, 1), self.w1, self.b1)
            a2 = self.forward_propagation(a1, self.w2, self.b2)
            if np.argmax(a2) == label[i]:
                precision += 1
        print("acc：%f" % (100 * precision / 10000) + "%")


class Dataloader(object):
    def __init__(self):
        f = h5py.File("train.hdf5", 'r')
        self.train_x, self.train_y = f['image'][...], f['label'][...]
        f.close()

        # load test
        f = h5py.File("test.hdf5", 'r')
        self.test_x, self.test_y = f['image'][...], f['label'][...]
        f.close()

        print("train_x", self.train_x.shape, self.train_x.dtype)
        print("train_y", self.train_y.shape, self.train_y.dtype)

    def SpinAug(self):
        import random
        import scipy.ndimage as sndm
        train_x = self.train_x
        for i in range(train_x.shape[0]):
            spinAngle = random.uniform(-6, 6 )
            train_x[i] = sndm.rotate(train_x[i], spinAngle, cval=0.01, order=1, reshape=False)
        return train_x

    def getData(self):
        x_train = self.SpinAug().transpose(1, 2, 0).reshape(-1, 60000)
        x_test = self.test_x.transpose(1, 2, 0).reshape(-1, 10000)
        y_train = np.ones((10, 60000)) * 0.01
        for i in range(60000):
            y_train[self.train_y[i]][i] = 0.99
        y_test = self.test_y.reshape(-1, 1)

        x_train = x_train / 255 * 0.99 + 0.01
        x_test = x_test / 255 * 0.99 + 0.01

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    dl = MLP(784, 200, 10, 0.1)
    dataloader = Dataloader()
    for i in range(5):
        x_train, y_train, x_test, y_test = dataloader.getData()
        dl.train(x_train, y_train)

    dl.predict(x_test, y_test)