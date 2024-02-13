import socket
import pickle
import numpy as np
import pandas as pd
import torch.optim as optim

import torch
import torch.nn as nn
import argparse

import datetime
import sys
import time
from _thread import *
# 定义网络
class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()  # 添加激活函数

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        output, h_0 = self.gru(x, h_0)
        output = self.fc(output[:, -1, :])
        output = self.activation(output)  # 添加激活函数
        return output, h_0

# 定义学习任务的设置
def parser_args():
    parser = argparse.ArgumentParser(description='GRU Model Parameters')
    parser.add_argument('--timestep', type=int, default=4, help='Time step size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--feature_size', type=int, default=6, help='Number of features')
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of the hidden layer')
    parser.add_argument('--output_size', type=int, default=3, help='Size of the output layer')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='gru', help='Name of the model')
    parser.add_argument('--save_path', type=str, default='./{}.pth'.format('gru'), help='Path to save the best model')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
    args = parser.parse_args()
    return args

def train_model(model, x_train, y_train, config):
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    # 设置初始隐藏状态（可根据需要调整）
    # hidden = None

    # 进行本地训练
    for epoch in range(config.epochs):
        optimizer.zero_grad()

        # 手动清除隐藏状态
        hidden = None

        # 前向传播
        output, hidden = model(x_train, hidden)
        loss = criterion(output, y_train)

        # 反向传播和参数更新
        loss.backward(retain_graph=True)
        optimizer.step()

        print(f"Local epoch [{epoch + 1}/{config.epochs}], Loss: {loss.item()}")

    # 返回更新后的客户端模型参数
    return model

# 划分数据
def split_data(data, timestep,feature_size, target_size):
    dataX = []
    dataY = []
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep])
        dataY.append(data[index + timestep, :target_size])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    train_size = int(np.round(dataX.shape[0]))
    x_train = dataX[:train_size, :]
    y_train = dataY[:train_size, :]
    x_test = dataX[train_size:, :]
    y_test = dataY[train_size:, :]

    return x_train, y_train, x_test, y_test


# 数据加载
def load_data(data_path):
    dataset = pd.read_csv(data_path)
    return dataset

def encodeParams(parameter) :
    '''
    将参数序列化为字节流
    '''
    return pickle.dumps(parameter, -1)

def decodeParams(parameter) :
    '''
    对网络接收的序列化参数进行解码
    '''
    return pickle.loads(parameter)

def local_training(data):
    global current_round
    global buffer
    global local_model
    global args
    # global length
    global client_socket
    global_params = decodeParams(data)
    print("[{}] << Recive global parameters | size : {}".format(datetime.datetime.now(), sys.getsizeof(data)))
    local_model.load_state_dict(global_params)
    # client training -> begin
    start_time = time.time()
    local_model.train()
    train_model(local_model, x_train_tensor, y_train_tensor, args)
    end_time = time.time()
    # client training -> end
    #########################################
    # add code to evaluate local model here
    #########################################
    client_parameters = local_model.state_dict()
    print("< Global Epoch {} > [{}] || Train done : {}".format(current_round, datetime.datetime.now(), len(encodeParams(client_parameters))))
    print("< Training Time > : {} ".format(end_time - start_time))

    client_socket.sendall(encodeParams(client_parameters))
    print("[{}] >> Send Trained Parameters".format(datetime.datetime.now()))
    buffer = b''
    current_round += 1


if __name__ == '__main__':
    args = parser_args()

    data_path ='tvhl.csv'
    dataset = load_data(data_path)

    selected_features = ['cpu_use', 'gpu_use', 'mem_use', 'cpu_plan', 'gpu_plan', 'mem_plan']
    target_size = 3  # 三个目标列的大小
    selected_data = dataset[selected_features].values

    x_train,y_train, x_test, y_test = split_data(selected_data, args.timestep, args.feature_size,target_size)
    x_train_tensor = torch.Tensor(x_train)
    y_train_tensor = torch.Tensor(y_train)

    # 创建TCP客户端套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 连接到服务器
    # server_address = ('127.0.0.1', 8888)
    server_address = ('202.117.43.11', 8888)
    client_socket.connect(server_address)

    # variable set
    buffer = b''
    current_round = 1
    max_epoch = 50   ## 全局epoch，注意要和server中的一样
    local_model = GRU(args.feature_size, args.hidden_size, args.num_layers, args.output_size).to('cpu')
    length = len(pickle.dumps(local_model.state_dict()))
    print("Length of Parameters : ", length)
    # optimizer = optim.SGD(local_model.parameters(), lr=args.learning_rate)

    while True:
        while len(buffer) < length:
            data = client_socket.recv(length - len(buffer))
            if not data:
                break
            buffer += data
        if len(buffer) >= length:
            print("[{}] << Recive init parameters".format(datetime.datetime.now()))
            local_training(buffer)
        if(current_round > max_epoch):
            time.sleep(5)
            print("Final global epoch .....")
            break

    client_socket.close()
