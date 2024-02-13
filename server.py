import socket
import pickle
import torch.nn as nn
import argparse
from _thread import *
import datetime
import sys
import time
import threading
 
# 创建互斥锁
mutex = threading.Lock()

# 定义神经网络
class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        output, h_0 = self.gru(x, h_0)
        output = self.fc(output[:, -1, :])
        output = self.activation(output)
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
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='gru', help='Name of the model')
    parser.add_argument('--save_path', type=str, default='./{}.pth'.format('gru'), help='Path to save the best model')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
    args = parser.parse_args()
    return args

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

def create_client_thread(client_socket, addr):
    '''
    线程函数，用于处理与客户端的通信
    '''
    global client_count
    global learn_flag
    global global_model
    global cur_parameters
    global new_parameters
    global length

    _cur_parameters = cur_parameters
    buffer = b''

    print("[{}] >> Connected by : <{}|{}>".format(datetime.datetime.now(), addr[0], addr[1]))

    client_socket.sendall(encodeParams(_cur_parameters))

    print("[{}] >> Send Global Parameters to <{}|{}>".format(datetime.datetime.now(), addr[0], addr[1]))

    while True:
        try:
            if not learn_flag:
                while len(buffer) < length:
                    data = client_socket.recv(length - len(buffer))
                    if not data:
                        break
                    buffer += data
                if len(buffer) >= length:
                    print("[{}] << Recive Local Parameters From <{}|{}> | Size : {}".format(datetime.datetime.now(), addr[0],  addr[1], sys.getsizeof(buffer)))
                    client_parameters = decodeParams(buffer)
                    mutex.acquire()
                    ### 模型聚合 -> begin
                    for key, value in client_parameters.items():
                        if key not in new_parameters.keys():
                            new_parameters[key] = value.clone()
                        else:
                            new_parameters[key] += value
                    ### 模型聚合 -> end
                    print("<{} : {}> Parameters Are Added".format(addr[0], addr[1]))
                    client_count += 1
                    buffer = b''
                    mutex.release()
        except ConnectionResetError as e:
            print("[{}] >> Disconnected by : <{}|{}>".format(datetime.datetime.now(), addr[0], addr[1]))
            break

    print("Client Thread Done......")
    client_socket.close()

def global_learning():
    global client_count
    global max_client
    global max_epoch
    global current_round
    global learn_flag
    global client_sockets
    global cur_parameters
    global new_parameters
    global global_model
    
    while True:
        # print("client_count : {} , max_client : {}".format(client_count, max_client))
        if(client_count >= max_client):
            # print("Global_Learning Begin...")
            learn_flag = True
            print("[{}] || Global Epoch {} : Local parameters all selected".format(datetime.datetime.now(), current_round))
            # FedAvg
            for key in new_parameters.keys():
                new_parameters[key] /= client_count
            global_model.load_state_dict(new_parameters)
            #########################################
            # add code to evaluate global model here
            #########################################
            print("[{}] || Global Aggregation done.".format(datetime.datetime.now()))
            
            cur_parameters = global_model.state_dict()
            new_parameters = {}

            mutex.acquire()
            client_count = 0
            current_round += 1
            learn_flag = False
            mutex.release()

            if(current_round > max_epoch):
                time.sleep(5)
                print("Final global epoch .....")
                server_socket.close()
                break
            else:
                for client in client_sockets:
                    client.sendall(encodeParams(cur_parameters))

    print("global Thread Done .....")
    print(">>> CTRL C to exit <<<")

if __name__ == '__main__':

    args = parser_args()
    global_model = GRU(args.feature_size, args.hidden_size, args.num_layers, args.output_size)

    # variable set
    client_sockets = []
    max_client = 3           ## 设置客户端数量
    max_epoch = args.epochs  ## 设置全局epoch
    client_count = 0
    current_round = 1
    learn_flag = False

    cur_parameters = global_model.state_dict()
    length = len(pickle.dumps(cur_parameters))
    print("Length of Parameters : ", length)
    new_parameters = {}

    # 创建服务器套接字
    print("Server Start......")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # 绑定服务器地址和端口
    # server_address = ('127.0.0.1', 4321)
    server_address = ('202.117.43.11', 8888)
    server_socket.bind(server_address)
    # 开始监听
    server_socket.listen()

    # wait for connect client
    try:
        start_new_thread(global_learning, ())
        while(client_count < max_client):
            print("Wait for Client......")
            client_socket, addr = server_socket.accept()
            client_sockets.append(client_socket)
            start_new_thread(create_client_thread, (client_socket, addr))
    except Exception as e:
        print('ERROR : ', e)

    finally:
        server_socket.close()
