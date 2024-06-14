import librosa

import os
noise_dir="/home/xinying/xinying/separated/noise"
speech_dir="/home/xinying/xinying/separated/speech"
noise_files=[]
speech_files=[]
for noise in os.listdir(noise_dir):
    noise_files.append(noise)
for speech in os.listdir(speech_dir):
    speech_files.append(speech)

    import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Flatten, Dense

# 设置随机种子以确保可重复性
np.random.seed(42)

# 定义音频文件路径和标签
# noise_files = ['path_to_noise_1.wav', 'path_to_noise_2.wav', ...]
# speech_files = ['path_to_speech_1.wav', 'path_to_speech_2.wav', ...]
labels = [0] * len(noise_files) + [1] * len(speech_files)
noise_dir="/home/xinying/xinying/separated/noise"
speech_dir="/home/xinying/xinying/separated/speech"
waveform, sample_rate = librosa.load(os.path.join(speech_dir, speech_files[0]), sr=None,duration=10)
# 提取音频特征
# 提取音频特征并进行零填充
def extract_features(root, files, max_length=None):
    features = []
    for file in files:
        waveform, sample_rate = librosa.load(os.path.join(root, file), sr=None,duration=10)
        mfcc = librosa.feature.mfcc(waveform, sample_rate)
        
        # 进行零填充
        if max_length is not None and mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        features.append(np.array(mfcc.T))
    
    return np.array(features)

# 提取特征并进行标准化
noise_features = extract_features(noise_dir, noise_files, max_length=10*sample_rate)
speech_features = extract_features(speech_dir, speech_files,max_length=10*sample_rate)

features = np.concatenate((noise_features, speech_features), axis=0)

mean_list = []
std_list = []
for sample in features:
    sample_mean = np.mean(sample, axis=0)
    sample_std = np.std(sample, axis=0)
    mean_list.append(sample_mean)
    std_list.append(sample_std)

# 计算特征的总体均值和标准差
mean = np.mean(mean_list, axis=0).mean()
std = np.mean(std_list, axis=0).mean()

features = (features - mean) / std

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 将数据展平为MLP所需的输入形状
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)



# 构建MLP模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))