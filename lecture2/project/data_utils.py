import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
    """加载CIFAR的单个批次文件"""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')  # python3 加 encoding
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # reshape成图像格式
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(root):
    """加载整个CIFAR-10数据集"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_CIFAR_batch(os.path.join(root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
