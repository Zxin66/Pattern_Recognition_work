import numpy as np
import os,cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,hamming_loss

# 函数：解析标签文件
#  3223 (_sex  female) (_age  senior) (_race black) (_face smiling) (_prop '(hat ))
# 序号-性别-年龄-种族-表情-其他
# 性别：male 0 female 1 error -1
# 年龄：child 0 teen 1 adult 2 seinior 3 error -1
# 种族：white 0 black 1 asian 2 hispanic 3 error -1
# 表情： smiling 0 serious 1 funny 2 error -1
# -1：标签错误
def parse_labels(labels_path):
    labels = []
    with open(labels_path,"r") as file:
        for line in file:
            # parts = line.strip().split(' ')
            parts = [part for part in line.strip().split(' ') if part]
            attributes = [0] * 4

            if(parts[1]=='(_missing'):
                continue
            # 性别
            if(parts[2]=='male)'):
                attributes[0] = 0
            elif(parts[2]=='female)'):
                attributes[0] = 1
            else:
                attributes[0] = -1

            # 年龄
            if(parts[4]=='child)' or parts[4]=='chil)'):
                attributes[1] = 0
            elif(parts[4]=='teen)'):
                attributes[1] = 1
            elif(parts[4]=='adult)' or parts[4]=='adulte)'):
                attributes[1] = 2
            elif(parts[4]=='seinior)'):
                attributes[1] = 3
            else:
                attributes[1] = -1

            # 种族
            if(parts[6]=='white)' or parts[6]=='whitee)'):
                attributes[2] = 0
            elif(parts[6]=='black)'):
                attributes[2] = 1
            elif(parts[6]=='asian)'):
                attributes[2] = 2
            elif(parts[6]=='hispanic)'):
                attributes[2] = 3
            else:
                attributes[2] = -1

            # 表情
            if(parts[8]=='smiling)' or parts[8]=='smilin)'):
                attributes[3] = 0
            elif(parts[8]=='serious)'):
                attributes[3] = 1
            elif(parts[8]=='funny)'):
                attributes[3] = 2
            else:
                attributes[3] = -1

            labels.append(attributes)
    return labels        

# 读取人脸图像,提取图像特征
# input:    图片文件夹路径
# output:   所有图片预处理后整合的列表
# 预处理：   统一尺寸，删大小格式错误的图片，直方图均衡
def load_images(img_folder_path):
    images = []
    image_files = [f for f in os.listdir(img_folder_path) if os.path.isfile(os.path.join(img_folder_path, f))]
    for file in image_files:
        file_path = os.path.join(img_folder_path,file)
        with open(file_path,"rb") as fid:
            img = np.fromfile(fid, dtype=np.uint8)   
        img_mat = img.reshape(128,128)  # 统一128*128尺寸
        img_mat = cv2.equalizeHist(img_mat) # 直方图均衡
        img_flattened = img_mat.flatten() # 展平为一维数组
        images.append(img_flattened)
        # plt.imshow(I_reshaped,cmap='gray')
        # plt.show()
    images = np.array(images)
    return images

# PCA函数：主成成分特征提取
# input     图像的二维矩阵（images）
# output    pca处理后的特征数据,pca
def algorithm_PCA(images):
    # 创建pca对象
    pca = PCA(n_components=150,whiten=True)
    pca.fit(images)
    features = pca.transform(images)
    return features, pca

# 训练SVM模型
# svm只能处理二进制的数据，需要先二值化
def train_svm(features, labels):
    classifiers = []
    for i in range(labels.shape[1]):  # 遍历每个属性
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(labels[:, i])  # 对每个属性进行二值化
        clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
        clf.fit(features, y_bin)  # 训练每个属性的分类器
        classifiers.append((lb, clf))  # 保存标签二值化器和分类器
    return classifiers

# 评估器
def evaluate_svm(classifiers,features,true_labels):
    results = {
        "precision": [],
        "recall": [],
        "f1": [],
        "hamming_loss": []
    }
    for i, (lb,clf) in enumerate(classifiers):
        true_labels_bin = lb.transform(true_labels[:,i])
        y_pred = clf.predict(features)

        precision = precision_score(true_labels_bin, y_pred, average=None)
        recall = recall_score(true_labels_bin, y_pred, average=None)
        f1 = f1_score(true_labels_bin, y_pred, average=None)
        hamming = hamming_loss(true_labels_bin, y_pred)

        results["precision"].append(precision[1])  # 只取正类的指标值
        results["recall"].append(recall[1])
        results["f1"].append(f1[1])
        results["hamming_loss"].append(hamming)
        
        # 打印当前维度的性能指标
        print(f"Dimension {i}:")
        print(f"Precision: {precision[1]:.3f}")
        print(f"Recall: {recall[1]:.3f}")
        print(f"F1 Score: {f1[1]:.3f}")
        print(f"Hamming Loss: {hamming:.3f}\n")
    
    return results

# 存放图像和标签的文件夹路径
img_file_path = 'face/rawdata'
labels_faceDR_path='face/faceDR'
labels_faceDS_path='face/faceDS'

poor_img = [2412,2416]#1188 1191
# 合并标签，删除大小错误的图片标签
labels_DR = parse_labels(labels_faceDR_path)
labels_DS = parse_labels(labels_faceDS_path)
labels = labels_DR + labels_DS
labels.pop(1188) 
labels.pop(1191)
labels = np.array(labels)
print('labels loading finished')

images = load_images(img_file_path)
print('images loading finished')

features,pca = algorithm_PCA(images)
print('features loading finished')

# 训练SVM模型
x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size=0.2,random_state=42)
classifiers = train_svm(x_train, y_train)
print('models loading finished')

# 评估器
evaluation_results = evaluate_svm(classifiers, x_test, y_test)