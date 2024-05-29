import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score,recall_score,f1_score,log_loss,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
class LogisticRegression:
    def __init__(self, learning_rate=0.002, num_iterations=8000, threshold=0.5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.threshold = threshold

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > self.threshold else 0 for i in y_predicted]
        return y_predicted_cls

    def fit(self, X, y ,x_val , y_val):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        loss_train = []  # 储存loss值
        loss_val = []  # 储存loss值

        # 梯度下降优化权重
        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            linear_model = np.dot(X, self.weights) + self.bias

            # 计算梯度
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            y_pred = self.predict(X)  # 01描述的预测结果
            y_pred_val = self.predict(x_val)
            cross_loss = log_loss(y, y_pred)
            cross_loss_val = log_loss(y_val, y_pred_val)
            if i > 10:
                loss_train.append(cross_loss)
                loss_val.append(cross_loss_val)
            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            # 打印迭代过程信息
            if i % 100 == 0:
                print(f"Iteration: {i}, Train Loss: {cross_loss}, Val Loss: {cross_loss_val}")
        return loss_train, loss_val

# 加载CSV文件
data = pd.read_csv("output_下采样_1-2.csv")
data_test = pd.read_csv("output_test_下采样.csv")
data_val = pd.read_csv("output_val.csv")
# 假设CSV文件的最后一列是标签列，其他列是特征列
X_train = data.iloc[:, :-1].values  # 特征
y_train = data.iloc[:, -1].values   # 标签
x_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].values
X_val = data_val.iloc[:, :-1].values
y_val = data_val.iloc[:, -1].values
#-----------------------------------------------------
threshold = 0.44
# 实例化逻辑回归模型
model = LogisticRegression(threshold=threshold)

# 使用fit方法拟合模型
loss_train, loss_val = model.fit(X_train, y_train, X_val , y_val)

predictions = model.predict(X_val)

Acc = accuracy_score(y_val, predictions)
Recall = recall_score(y_val, predictions)
f1 = f1_score(y_val, predictions)

print("训练集")
print("Accuracy:{:.2f}%".format(Acc*100))
print("Recall:{:.2f}%".format(Recall * 100))
print("F1 score:{:.2f}%".format(f1 * 100))

plt.figure()
plt.plot(loss_train, label="train_loss")
plt.plot(loss_val, label="val_loss")
plt.ylabel("cross entropy")
plt.xlabel("epoch")
plt.legend()
plt.title("loss_train")
plt.show()

# test
predictions_test = model.predict(x_test)
Acc = accuracy_score(y_test, predictions_test)
Recall = recall_score(y_test, predictions_test)
f1 = f1_score(y_test, predictions_test)


print("测试集")
print("Accuracy:{:.2f}%".format(Acc * 100))
print("Recall:{:.2f}%".format(Recall * 100))
print("F1 score:{:.2f}%".format(f1 * 100))

cm = confusion_matrix(y_true=y_test, y_pred=predictions_test)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

# 设置坐标轴标签
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Predicted Negative', 'Predicted Positive'], rotation=45)
plt.yticks(tick_marks, ['Actual Negative', 'Actual Positive'])

# 添加数据标签
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

# 添加坐标轴标签
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
# model就是训练完的模型了

#
