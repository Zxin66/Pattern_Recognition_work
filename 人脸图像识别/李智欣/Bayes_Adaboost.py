import os
import cv2
import numpy as np
import skimage.feature as sk
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

def parse_labels(labels_path):
    init_labels = {}
    with open(labels_path, "r") as file:
        for line in file:
            parts = [part for part in line.strip().split(' ') if part]
            if len(parts) < 9 or parts[1] == '(_missing)':
                continue  # 跳过不完整的行或缺失的图像
            seq_num = parts[0]  # 图像编号作为键
            attributes = [0] * 4  # 初始化属性列表

            # 性别
            if parts[2] == 'male)':
                attributes[0] = 0
            elif parts[2] == 'female)':
                attributes[0] = 1
            else:
                attributes[0] = -1  # 错误的性别标签

            # 年龄
            if parts[4] in ('child)', 'chil)'):
                attributes[1] = 0
            elif parts[4] == 'teen)':
                attributes[1] = 1
            elif parts[4] in ('adult)', 'adulte)'):
                attributes[1] = 2
            # elif parts[4] == 'seinior)':
            elif parts[4] in ('seinior)', 'senior)'):
                attributes[1] = 3
            else:
                attributes[1] = -1  # 错误年龄标签

            # 种族
            if parts[6] in ('white)', 'whitee)'):
                attributes[2] = 0
            elif parts[6] == 'black)':
                attributes[2] = 1
            elif parts[6] == 'asian)':
                attributes[2] = 2
            elif parts[6] == 'hispanic)':
                attributes[2] = 3
            else:
                attributes[2] = -1  # 错误的种族标签

            # 表情
            if parts[8] in ('smiling)', 'smilin)'):
                attributes[3] = 0
            elif parts[8] == 'serious)':
                attributes[3] = 1
            elif parts[8] == 'funny)':
                attributes[3] = 2
            else:
                attributes[3] = -1  # 错误的表情标签

            init_labels[seq_num] = attributes  # 将属性列表与图像编号相关联
    return init_labels

def match_images_and_labels(image_folder, labels_dict):
    images_information = []
    labels_change = []
    gender_labels = []
    age_labels = []
    race_labels = []
    expression_labels = []
    filenames = []
    # print("start to match images and labels..........")
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    for file in image_files:
        file_path = os.path.join(image_folder,file)
        with open(file_path,"rb") as fid:
            img = np.fromfile(fid, dtype=np.uint8)
        if img is not None:
            seq_num = file
            if seq_num in labels_dict:
                label = labels_dict[seq_num]
                images_information.append(img)
                labels_change.append(label)
                filenames.append(file)

    labels_change = np.array(labels_change)
    for index in range(len(labels_change)):
        gender_labels.append(labels_change[index][0])
        age_labels.append(labels_change[index][1])
        race_labels.append(labels_change[index][2])
        expression_labels.append(labels_change[index][3])

    # print("finish match")
    return images_information, gender_labels, age_labels, race_labels, expression_labels

def pca_feature(images, n_components):
    img_flatten = np.array([img.flatten() for img in images])  #展平
    # 创建PCA对象并设置要保留的主成分数量
    pca = PCA(n_components=n_components)
    # 对展平后的图像数据应用PCA进行特征提取和降维
    pca_features = pca.fit_transform(img_flatten)
    return pca_features  # 将二维的特征矩阵（n_samples, n_components）转换为一维向量返回

def lbp(image, n, r):
    img = image.astype(np.uint8)
    lbp_image = sk.local_binary_pattern(img, n, r, method='uniform')
    # 将LBP图像转换为直方图
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n+10), range=(0, n+10), density=True)
    # 归一化直方图
    hist = hist / hist.sum()
    return hist

def extrat_features(images_arr):
    # print("start feature.......")
    feature = []
    img_arr = []
    for i in range(len(images_arr)):
        # 重新塑形为 128x128 的矩阵
        if len(images_arr[i]) == 16384:
            img = images_arr[i].reshape(128, 128)
        if len(images_arr[i]) == 262144:
            img_reshaped = images_arr[i].reshape(512, 512)
            # 计算裁剪的起始位置，以获取中心 128x128 的区域
            center_x = img_reshaped.shape[1] // 2
            center_y = img_reshaped.shape[0] // 2
            x_start = center_x - 64  # 128/2
            y_start = center_y - 64  # 128/2

            # 裁剪中心的 128x128 区域
            cropped_image = img_reshaped[y_start:y_start + 128, x_start:x_start + 128]
            img = resize(cropped_image, (128, 128), anti_aliasing=True)
        img_arr.append(img)
        # 设置LBP参数
        # radius = 1
        # n_points = 10 * radius
        # hist = lbp(img, n_points, radius)
        # feature.append(hist)
        # pca_feature(img, n_components=100)
        # feature.append(pca_feature)

    img_ = np.array(img_arr)
    print(len(img_))
    features = pca_feature(img_, n_components=150)
    return features

def print_result(labels, pred):
    print("total:", accuracy_score(labels, pred))
    print("macro average")
    print("precision:",precision_score(labels, pred, average='macro'))
    print("recall:", recall_score(labels, pred, average='macro'))
    print("f1:",f1_score(labels, pred, average='macro'))
    print("weighted average")
    print("precision:",precision_score(labels, pred, average='weighted'))
    print("recall:", recall_score(labels, pred, average='weighted'))
    print("f1:",f1_score(labels, pred, average='weighted'))

#性别
def gender_bys_ada(gender_label, features_arr, k, n):
    print("start gender_bys_ada......")
    features_arr = np.array(features_arr)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    all_accuracies = []
    for train_index, test_index in skf.split(features_arr, gender_label):
        # 划分训练集和测试集特征数据
        x_train_gender, x_test_gender = features_arr[train_index], features_arr[test_index]
        # 划分训练集和测试集标签数据
        y_train_gender, y_test_gender = np.array(gender_label)[train_index], np.array(gender_label)[test_index]

        # 创建Adaboost分类器对象，使用高斯朴素贝叶斯作为弱分类器
        gender_clf = AdaBoostClassifier(GaussianNB(), n_estimators=n, random_state=42)
        # 使用划分好的训练集数据训练模型
        gender_clf.fit(x_train_gender, y_train_gender)
        # 使用训练好的模型对测试集进行预测
        gender_pred = gender_clf.predict(x_test_gender)
        # 计算本次折叠的准确率并添加到列表中
        accuracy = accuracy_score(y_test_gender, gender_pred)
        all_accuracies.append(accuracy)

    # 计算平均准确率
    average_accuracy = np.mean(all_accuracies)
    print("gender average(fold={}):".format(k), average_accuracy)

    #在全部数据上重新训练模型
    final_gender_clf = AdaBoostClassifier(GaussianNB(), n_estimators=n, random_state=42)
    final_gender_clf.fit(features_arr, np.array(gender_label))
    final_gender_pred = final_gender_clf.predict(features_arr)

    gender_label = np.array(gender_label)
    print_result(gender_label, final_gender_pred)
#年龄
def age_bys_ada(age_label, features_arr,k, n):
    print("start age_bys_ada......")
    features_arr = np.array(features_arr)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_accuracies = []
    for train_index, test_index in kf.split(features_arr):
        x_train_age, x_test_age = features_arr[train_index], features_arr[test_index]
        y_train_age, y_test_age = np.array(age_label)[train_index], np.array(age_label)[test_index]

        age_clf = AdaBoostClassifier(GaussianNB(), n_estimators=n, random_state=42)
        age_clf.fit(x_train_age, y_train_age)
        age_pred = age_clf.predict(x_test_age)

        accuracy = accuracy_score(y_test_age, age_pred)
        all_accuracies.append(accuracy)

    # 计算平均准确率
    average_accuracy = np.mean(all_accuracies)
    print("age average(fold={}):".format(k), average_accuracy)

    final_age_clf = AdaBoostClassifier(GaussianNB(), n_estimators=n, random_state=42)
    final_age_clf.fit(features_arr, np.array(age_label))
    final_age_pred = final_age_clf.predict(features_arr)
    age_label = np.array(age_label)
    print_result(age_label, final_age_pred)
    # print("total:", accuracy_score(np.array(age_label), final_age_pred))
    # print(classification_report(np.array(age_label), final_age_pred))
#种族
def race_bys_ada(race_label, features_arr,k, n):
    print("start race_bys_ada......")
    features_arr = np.array(features_arr)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_accuracies = []
    for train_index, test_index in kf.split(features_arr):
        x_train_race, x_test_race = features_arr[train_index], features_arr[test_index]
        y_train_race, y_test_race = np.array(race_label)[train_index], np.array(race_label)[test_index]

        race_clf = AdaBoostClassifier(GaussianNB(), n_estimators=n, random_state=42)
        race_clf.fit(x_train_race, y_train_race)
        race_pred = race_clf.predict(x_test_race)

        accuracy = accuracy_score(y_test_race, race_pred)
        all_accuracies.append(accuracy)

    # 计算平均准确率
    average_accuracy = np.mean(all_accuracies)
    print("race average(fold={}):".format(k), average_accuracy)

    final_race_clf = AdaBoostClassifier(GaussianNB(), n_estimators=n, random_state=42)
    final_race_clf.fit(features_arr, np.array(race_label))
    final_race_pred = final_race_clf.predict(features_arr)

    race_label = np.array(race_label)
    print_result(race_label, final_race_pred)
    # print("total:", accuracy_score(np.array(race_label), final_race_pred))
    # print(classification_report(np.array(race_label), final_race_pred))
#表情
def expression_bys_ada( expression_label, features_arr, k, n):
    print("start expression_bys_ada......")
    features_arr = np.array(features_arr)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_accuracies = []
    for train_index, test_index in kf.split(features_arr):
        x_train_expression, x_test_expression = features_arr[train_index], features_arr[test_index]
        y_train_expression, y_test_expression = np.array(expression_label)[train_index], np.array(expression_label)[test_index]

        expression_clf = AdaBoostClassifier(GaussianNB(), n_estimators=n, random_state=42)
        expression_clf.fit(x_train_expression, y_train_expression)
        expression_pred = expression_clf.predict(x_test_expression)

        accuracy = accuracy_score(y_test_expression, expression_pred)
        all_accuracies.append(accuracy)

    # 计算平均准确率
    average_accuracy = np.mean(all_accuracies)
    print("expression average(fold={}:".format(k), average_accuracy)

    final_expression_clf = AdaBoostClassifier(GaussianNB(), n_estimators=n, random_state=42)
    final_expression_clf.fit(features_arr, np.array(expression_label))
    final_expression_pred = final_expression_clf.predict(features_arr)

    expression_label = np.array(expression_label)
    print_result(expression_label, final_expression_pred)
    # print("total:", accuracy_score(np.array(expression_label), final_expression_pred))
    # print(classification_report(np.array(expression_label), final_expression_pred))

labels_path_faceDR = r'D:\subject learning\专业选修\模式识别\人脸图像识别\face\faceDR'
labels_path_faceDS = r'D:\subject learning\专业选修\模式识别\人脸图像识别\face\faceDR'
labels_dict_DR = parse_labels(labels_path_faceDR)
labels_dict_DS = parse_labels(labels_path_faceDS)
combine_labels_dict = {**labels_dict_DR, **labels_dict_DS}
# print("combine dict has ready")

image_path = r'D:\subject learning\专业选修\模式识别\人脸图像识别\face\rawdata'
images_information, gender_labels, age_labels, race_labels, expression_labels = match_images_and_labels(image_path ,combine_labels_dict)
features = extrat_features(images_information)

k = 15
n=50
gender_bys_ada(gender_labels, features,k, n)
age_bys_ada(age_labels, features,k, n)
race_bys_ada(race_labels, features,k, n)
expression_bys_ada(expression_labels, features,k, n)
