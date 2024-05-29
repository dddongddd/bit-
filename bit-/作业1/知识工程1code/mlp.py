import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv("output_下采样_1-2.csv")
data_test = pd.read_csv("output_test_下采样.csv")
data_val = pd.read_csv("output_val.csv")

# 数据预处理
X_train = data.iloc[:, :-1].values  # 特征
y_train = data.iloc[:, -1].values   # 标签
X_test = data_test.iloc[:, :-1].values
y_test = data_test.iloc[:, -1].values
X_val = data_val.iloc[:, :-1].values
y_val = data_val.iloc[:, -1].values

# 定义MLP模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # 输入层
    tf.keras.layers.Dense(16, activation='relu'),  # 隐藏层
    tf.keras.layers.Dense(16, activation='relu'),  # 隐藏层
    tf.keras.layers.Dense(16, activation='relu'),  # 隐藏层
    tf.keras.layers.Dense(1, activation='sigmoid')  # 输出层
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_val, y_val))

# 评估模型
predictions = model.predict(X_val)
predictions = (predictions > 0.4).astype(int)

Acc = accuracy_score(y_val, predictions)
Recall = recall_score(y_val, predictions)
f1 = f1_score(y_val, predictions)

print("验证集")
print("Accuracy:{:.2f}%".format(Acc*100))
print("Recall:{:.2f}%".format(Recall * 100))
print("F1 score:{:.2f}%".format(f1 * 100))

# 绘制训练过程中的损失和精度变化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 测试集上的性能
predictions_test = model.predict(X_test)
predictions_test = (predictions_test > 0.5).astype(int)

Acc = accuracy_score(y_test, predictions_test)
Recall = recall_score(y_test, predictions_test)
f1 = f1_score(y_test, predictions_test)

print("测试集")
print("Accuracy:{:.2f}%".format(Acc * 100))
print("Recall:{:.2f}%".format(Recall * 100))
print("F1 score:{:.2f}%".format(f1 * 100))

# 绘制混淆矩阵
cm = confusion_matrix(y_true=y_test, y_pred=predictions_test)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Predicted Negative', 'Predicted Positive'], rotation=45)
plt.yticks(tick_marks, ['Actual Negative', 'Actual Positive'])

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
