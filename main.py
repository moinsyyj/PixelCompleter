import math
import os
import pickle
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


def img2Y(img):
    Y = np.empty([64, 24 * 21])
    col = 0
    for height in range(0, 192, 8):
        for width in range(0, 168, 8):
            Y[:, col] = img[height:height + 8, width:width + 8].reshape(64)
            col += 1
    return Y


def Y2img(Y):
    img = np.empty([192, 168])
    col = 0
    for height in range(0, 192, 8):
        for width in range(0, 168, 8):
            img[height:height + 8, width:width + 8] = Y[:, col].reshape(8, 8)
            col += 1
    return img


def k_svd(Y, K_column, max_iter=30, threshold=0.1, n_nonzero_coefs=15):
    """
    Y = DX
    """
    D = np.random.rand(64, K_column) / np.sqrt(K_column)  # 初始化字典
    err = []  # 记录训练集平均误差
    for _iter in range(max_iter):
        X = omp(D, Y, n_nonzero_coefs)  # OMP，得到Y的稀疏表达X
        # X = linear_model.orthogonal_mp(D, Y, n_nonzero_coefs=n_nonzero_coefs)
        _err = np.abs(Y - np.dot(D, X)).mean() * 255.0
        print(f'INFO: {_iter} iter, MAE={_err}')
        err.append(_err)
        if _err < threshold:
            break
        # 更新字典
        for k in range(K_column):
            # 取出稀疏编码X中，第k行不为0的列
            indexes = np.nonzero(X[k, :])[0]
            if len(indexes) > 0:  # 如果这一列全为0，无法计算
                D[:, k] = 0
                R = (Y - np.dot(D, X))[:, indexes]
                # 利用svd的方法，来求解更新字典和稀疏系数矩阵
                u, s, v = np.linalg.svd(R, full_matrices=False)
                # 使用左奇异矩阵的第0列更新字典
                D[:, k] = u[:, 0]
                # 使用第0个奇异值和右奇异矩阵的第0行的乘积更新稀疏系数矩阵
                X[k, indexes] = s[0] * v[0]
    plt.plot(err)
    plt.show()
    return D


def omp(D, Y, n_nonzero_coefs):
    if len(Y.shape) == 1:  # 单列
        Y = Y.reshape((Y.shape[0], 1))  # 二维矩阵
    X = np.zeros((D.shape[1], Y.shape[1]))  # Y=DX
    for _col in range(Y.shape[1]):
        y = Y[:, _col]
        r = y  # 残差
        x = None  # 基向量长度x i
        d = None  # 临时矩阵
        indexes = []
        for _ in range(n_nonzero_coefs):  # 稀疏度
            proj = np.fabs(np.dot(D.T, r))  # 最大投影
            pos = np.argmax(proj)
            indexes.append(pos)
            if _ == 0:
                d = D[:, pos].reshape(Y.shape[0], 1)
            else:
                d = np.append(d, D[:, pos].reshape(Y.shape[0], 1), axis=1)
            x = np.dot(np.linalg.pinv(d), y)
            r = y - np.dot(d, x)
        X[indexes, _col] = x
    return X


def recover(D, img):
    Y_input = img2Y(img) / 255.0
    for i in range(504):
        img_col = Y_input[:, i]
        indexes = np.nonzero(img_col)[0]
        if len(indexes) > 0:  # 如果这一列全为0，无法计算
            x = omp(D[indexes], img_col[indexes], 10)
            # x = linear_model.orthogonal_mp(D[indexes], img_col[indexes], n_nonzero_coefs=10)
            Y_input[:, i] = np.dot(D, x).reshape(64)
    return Y2img(Y_input) * 255.0


if __name__ == '__main__':
    # 预置数据
    HEIGHT = 192
    WIDTH = 168
    DATA_SIZE = 38
    # 可变数据
    should_train = False  # 是否训练新字典
    loss_rate = 0.3  # 测试图片损失率
    K = 441  # 字典D的维数 64*K
    # 读取数据集
    ALL_DATA = np.empty([DATA_SIZE, HEIGHT, WIDTH], dtype=np.uint8)
    for root, dirs, files in os.walk('./DataSet'):
        for _ in range(DATA_SIZE):
            ALL_DATA[_] = np.fromfile(os.path.join(root, files[_]), dtype=np.uint8,
                                      count=HEIGHT * WIDTH, offset=15).reshape(192, 168)
    # 分割数据集
    TRAIN_SIZE = math.floor(DATA_SIZE * 0.8)
    TRAIN_DATA = ALL_DATA[:TRAIN_SIZE]
    print('TRAIN_DATA', TRAIN_DATA.shape)
    TEST_DATA = ALL_DATA[TRAIN_SIZE:]
    print('TEST_DATA', TEST_DATA.shape)
    # 训练字典
    if should_train:
        # 随机选取训练集[0,idx) 8x8小块
        Y_train = np.empty([64, 10980])
        col = 0
        for _img in TRAIN_DATA:  # 这是一个危险的随机选取方法
            indexes = np.random.randint(0, 504, 366)  # 每张图随机取10980/30=366块
            temp_y = img2Y(_img)
            for _ in indexes:
                Y_train[:, col] = temp_y[:, _]
                col += 1
        Y_train = Y_train / 255.0  # 归一化。或用L2范数归一化
        print('Y', Y_train.shape)
        print("INFO: 即将开始字典学习。")
        my_dict = k_svd(Y_train, K, max_iter=50, threshold=0.1, n_nonzero_coefs=10)
        with open('KSVD_dict', 'wb') as f:
            pickle.dump(my_dict, f)
        print("INFO: 字典学习结束，已保存到KSVD_dict。")
    else:
        with open('KSVD_dict', 'rb') as f:
            my_dict = pickle.load(f)
        print("INFO: 成功读取字典，即将开始图片复原。")
    # 使用字典
    img_inputs = TEST_DATA.copy()
    for _ in range(img_inputs.shape[0]):
        img_inputs[_][np.random.rand(HEIGHT, WIDTH) <= loss_rate] = 0  # 像素缺失处理
        img_rec = recover(my_dict, img_inputs[_])  # 恢复像素
        err = np.abs(TEST_DATA[_] - img_rec).mean()
        print(f'INFO: 图片{_}修复完毕, MAE={err}')
        # 作图
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        plt.axis('off')
        ax1.imshow(TEST_DATA[_], cmap='gray')
        ax2 = fig.add_subplot(132)
        plt.axis('off')
        ax2.imshow(img_inputs[_], cmap='gray')
        ax3 = fig.add_subplot(133)
        plt.axis('off')
        ax3.imshow(img_rec, cmap='gray')
        plt.show()
        plt.close()
