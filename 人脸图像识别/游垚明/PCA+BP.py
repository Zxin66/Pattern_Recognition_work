import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, accuracy_score

# 忽略 ConvergenceWarning 警告
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 读取人脸图像,提取图像特征
# input:    图片文件夹路径
# output:   所有图片预处理后整合的列表,数据大小错误的文件名
# 预处理：   统一尺寸，删大小格式错误的图片
def load_images(img_folder_path):
    images = []
    error = []  # 存储错误文件信息
    image_files = [f for f in os.listdir(img_folder_path) if os.path.isfile(os.path.join(img_folder_path, f))]
    for file in image_files:
        file_path = os.path.join(img_folder_path, file)
        with open(file_path, "rb") as fid:
            img = np.fromfile(fid, dtype=np.uint8)
        if (img.size != 16384):  # 删大小格式错误的图片
            error.append(str(file))
            continue
        img_mat = img.reshape(128, 128)  # 统一128*128尺寸
        img_flattened = img_mat.flatten()  # 展平为一维数组
        # plt.imshow(img_mat,cmap='gray')
        # plt.show()
        images.append(img_flattened)
    images = np.array(images)

    return images, error

# 函数：解析标签文件
# 序号-性别-年龄-种族-表情-其他
# 性别：male 0 female 1
# 年龄：child 0 teen 1 adult 2 seinior 3
# 种族：white 0 black 1 asian 2 hispanic 3
# 表情： smiling 0 serious 1 funny 2
# -1：标签错误
def parse_labels(labels_path, error):
    labels = []
    with open(labels_path, "r") as file:
        for line in file:
            parts = [part for part in line.strip().split(' ') if part]
            attributes = [0] * 4

            if (parts[1] == '(_missing'):  # 删除丢失数据标签
                continue

            if (parts[0] in error):  # 删除错误数据标签
                continue

            # 性别
            if (parts[2] == 'male)'):
                attributes[0] = 0
            elif (parts[2] == 'female)'):
                attributes[0] = 1
            else:
                attributes[0] = -1

            # 年龄
            if (parts[4] == 'child)' or parts[4] == 'chil)'):
                attributes[1] = 0
            elif (parts[4] == 'teen)'):
                attributes[1] = 1
            elif (parts[4] == 'adult)' or parts[4] == 'adulte)'):
                attributes[1] = 2
            elif (parts[4] == 'seinior)'):
                attributes[1] = 3
            else:
                attributes[1] = -1

            # 种族
            if (parts[6] == 'white)' or parts[6] == 'whitee)'):
                attributes[2] = 0
            elif (parts[6] == 'black)'):
                attributes[2] = 1
            elif (parts[6] == 'asian)'):
                attributes[2] = 2
            elif (parts[6] == 'hispanic)'):
                attributes[2] = 3
            else:
                attributes[2] = -1

            # 表情
            if (parts[8] == 'smiling)' or parts[8] == 'smilin)'):
                attributes[3] = 0
            elif (parts[8] == 'serious)'):
                attributes[3] = 1
            elif (parts[8] == 'funny)'):
                attributes[3] = 2
            else:
                attributes[3] = -1

            labels.append(attributes)
    return labels


# PCA函数：主成成分特征提取
# input     图像的二维矩阵（images）
# output    pca处理后的特征数据
def algorithm_PCA(images):
    pca = PCA(n_components=150, whiten=True, svd_solver="randomized")  # 150个特征
    pca.fit(images)
    features = pca.transform(images)
    return features


def BP(labels, features):
    list_labels = ['性别', '年龄', '种族', '表情']  # 确保这是正确的标签列表
    for i in range(labels.shape[1]):
        y_label = labels[:, i]
        X_train, X_test, y_train, y_test = train_test_split(features, y_label, test_size=0.3, random_state=1,
                                                            stratify=y_label)
        # 神经网络数据缩放
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1, max_iter=2000,
                            early_stopping=True,
                            learning_rate_init=0.1, learning_rate='invscaling', power_t=0.5, activation="logistic")

        # 应用 K 折交叉验证
        scores = cross_val_score(clf, X_train_std, y_train, cv=10)  # 使用 10 折交叉验证
        print(f"{list_labels[i]} Accuracy: {scores.mean():.2f}")

        clf.fit(X_train_std, y_train)  # 训练每个属性的分类器
        y_pred = clf.predict(X_train_std)
        precision = precision_score(y_train, y_pred, average="weighted")
        recall = recall_score(y_train, y_pred, average="weighted")
        f1 = f1_score(y_train, y_pred, average="weighted")
        # 打印当前维度的性能指标

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
    return 0
#输入路径
img_file_path = 'rawdata'
labels_faceDR_path = 'faceDR'
labels_faceDS_path = 'faceDS'

# 图像处理
images, error = load_images(img_file_path)

# 合并标签
labels_DR = parse_labels(labels_faceDR_path, error)
labels_DS = parse_labels(labels_faceDS_path, error)
labels = labels_DR + labels_DS
labels = np.array(labels)
# pca
features = algorithm_PCA(images)

BP(labels, features)