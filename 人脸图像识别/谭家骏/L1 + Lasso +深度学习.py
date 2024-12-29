import numpy as np
import os
import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def parse_labels(labels_path):
    labels = []
    with open(labels_path, "r") as file:
        for line in file:
            parts = [part for part in line.strip().split(' ') if part]
            attributes = [0] * 4

            if (parts[1] == '(_missing'):
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
def load_images(img_folder_path):
    images = []
    image_files = [f for f in os.listdir(img_folder_path) if os.path.isfile(os.path.join(img_folder_path, f))]
    for file in image_files:
        file_path = os.path.join(img_folder_path, file)
        with open(file_path, "rb") as fid:
            img = np.fromfile(fid, dtype=np.uint8)
        img_mat = img.reshape(128, 128)  # 统一128*128尺寸
        img_mat = cv2.equalizeHist(img_mat)  # 直方图均衡
        img_flattened = img_mat.flatten()  # 展平为一维数组
        images.append(img_flattened)
    images = np.array(images)
    return images

# 自定义数据集类
class FaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 定义深度神经网络
class FaceClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FaceClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.l1_lambda = 0.01

    def forward(self, x):
        return self.layers(x)

    def l1_regularization(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss


# Lasso特征选择
def select_features(X, y, n_features=5000):
    lasso = Lasso(alpha=0.005, max_iter=10000)
    lasso.fit(X, y)
    importance = np.abs(lasso.coef_)
    selected_indices = np.argsort(importance)[-n_features:]
    return selected_indices


# 训练单个模型
def train_single_model(X_train, y_train, X_test, y_test, num_classes, device):
    # 创建数据加载器
    train_dataset = FaceDataset(X_train, y_train)
    test_dataset = FaceDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # 初始化模型
    model = FaceClassifier(X_train.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    # 训练
    best_acc = 0
    for epoch in range(500):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y) + model.l1_regularization()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total
        if accuracy > best_acc:
            best_acc = accuracy

        scheduler.step(total_loss)

    return best_acc


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    img_folder_path = 'face/rawdata'
    labels_faceDR_path = 'face/faceDR'
    labels_faceDS_path = 'face/faceDS'
    labels_DR = parse_labels(labels_faceDR_path)
    labels_DS = parse_labels(labels_faceDS_path)
    labels = labels_DR + labels_DS
    labels.pop(1188)
    labels.pop(1191)
    # 加载图像和标签
    images = load_images(img_folder_path)
    labels = np.array(labels)

    # 数据标准化
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)

    # 任务相关参数
    tasks = ['性别', '年龄', '种族', '表情']
    num_classes = [2, 4, 4, 3]

    # K折交叉验证
    k_fold = KFold(n_splits=10, shuffle=True, random_state=42)

    # 对每个任务进行训练和评估
    for task_idx, (task, n_classes) in enumerate(zip(tasks, num_classes)):
        print(f"\n开始训练 {task} 分类器...")

        # 获取当前任务的标签
        y = labels[:, task_idx]

        # 移除无效标签的样本
        valid_mask = y != -1
        X_valid = images_scaled[valid_mask]
        y_valid = y[valid_mask]

        # 特征选择
        selected_features = select_features(X_valid, y_valid)
        X_selected = X_valid[:, selected_features]

        # K折交叉验证
        fold_accuracies = []
        for fold, (train_idx, test_idx) in enumerate(k_fold.split(X_selected)):
            print(f"正在训练第 {fold + 1} 折...")

            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y_valid[train_idx], y_valid[test_idx]

            # 训练和评估模型
            accuracy = train_single_model(X_train, y_train, X_test, y_test, n_classes, device)
            fold_accuracies.append(accuracy)
            print(f"第 {fold + 1} 折准确率: {accuracy:.4f}")

        # 输出平均准确率
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        print(f"{task} 分类器的平均准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")


if __name__ == "__main__":
    main()