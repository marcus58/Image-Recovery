from matplotlib import pyplot as plt  # 展示图片
import numpy as np  # 数值处理
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析

def noise_mask_image(img, noise_ratio):
    """
    根据题目要求生成受损图片
    :param img: 图像矩阵，一般为 np.ndarray
    :param noise_ratio: 噪声比率，可能值是0.4/0.6/0.8
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array, 
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None

    # -------------实现受损图像答题区域-----------------
    R = np.random.random(img.shape)
    I = np.array(R >= noise_ratio, dtype='double')
    M, N, C = img.shape
    while True:
        total = I.sum() / (M * N * C)
        if abs(1 - total - noise_ratio) <= 1e-6:
            break
        R = np.random.random(img.shape)
        I = np.array(R >= noise_ratio, dtype='double')
    noise_img = img * I
    # -----------------------------------------------

    return noise_img
    
def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')

from PIL import Image
import numpy as np

def CalDis(i, j, ci, cj):
    return max(ci - i, i - ci) + max(cj - j, j - cj)

def LocalValue(i, j, imgSlide, mskSlide, size):
    cursize = size
    maxsize = 8
    while cursize <= maxsize:
        rowU = max(i - cursize, 0)
        rowD = min(i + cursize, imgSlide.shape[0] - 1)

        colL = max(j - cursize, 0)
        colR = min(j + cursize, imgSlide.shape[1] - 1)

        y = []
        num=[]
        totalDis = 0
        for ci in range(rowU, rowD + 1):
            for cj in range(colL, colR + 1):
                num.append(imgSlide[ci, cj])
                if mskSlide[ci, cj] != 0.0:
                    dis = cursize * 2 - CalDis(i, j, ci, cj) + 1  # Laplace
                    dis = dis ** 8
                    totalDis += dis
                    y.append(imgSlide[ci, cj] * dis)
        maxval = max(num)
        minval = min(num)
        y = np.array(y, ndmin=1)
        if y.shape[0] == 0:
            cursize += 1
        else:
            localvalue = y.sum()/totalDis
            if (localvalue > minval + (maxval-minval)*0.1 and localvalue < maxval - (maxval-minval)*0.1) or cursize==maxsize:
                #print(cursize)
                return localvalue
            else:
                cursize += 1

def Predict(i, j, imgSlide, mskSlide, localValue, size):
    rowD = min(i + size, imgSlide.shape[0] - 1) + 1
    colR = min(j + size, imgSlide.shape[1] - 1) + 1

    X = []
    y = []
    for ci in range(i, rowD):
        for cj in range(j, colR):
            # print(type(X))
            X.append([ci, cj])
            y.append(localValue[ci, cj])

    X = np.array(X, ndmin=2)
    y = np.array(y, ndmin=1)

    clf = LinearRegression()
    clf.fit(X, y)

    for ci in range(i, rowD):
        for cj in range(j, colR):
            if mskSlide[ci, cj] == 0.0:
                XPred = np.array([i, j], ndmin=2)
                imgSlide[i, j] = clf.predict(XPred)[0]
                imgSlide[i, j] = min(max(imgSlide[i, j], 0), 1)
    return rowD, colR, imgSlide[i:i+rowD, j:j+colR]

def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------
    for c in range(3):
        # print('slide =', c)
        imgSlide = res_img[:, :, c]
        mskSlide = noise_mask[:, :, c]
        localValue = res_img[:, :, c]

        M, N = imgSlide.shape
        for i in range(M):
            for j in range(N):
                if mskSlide[i, j] == 0.0:
                    # 求localValue
                    localValue[i, j] = LocalValue(i, j, imgSlide, mskSlide, 2)
                else:
                    localValue[i, j] = imgSlide[i, j]

        for i in range(0, M, size):
            for j in range(0, N, size):
                rowD, colR, temp = Predict(i, j, imgSlide, mskSlide, localValue, size)
                res_img[i:i+rowD, j:j+colR, c] = temp
    # ---------------------------------------------------------------

    return res_img