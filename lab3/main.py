import numpy as np 
import pandas as pd 
import os 

from utils.cal_roll_pitch_yaw import cal_orientation
from utils.forward_kinematics import cal_forward_kinematics
from utils.trainer import train_model, calculate_mae
from utils.dataset import RobotDataset
from models.rnn import RobotModel 
from genetic.genetic_al import gen_al
from genetic.functions import crossover, mutate
from genetic.initialization import initialize_population
from utils.generate_data import create_data









# define cac thong so cho truoc 
population_size = 100
num_generations = 100
mutation_rate = 0.01
crossover_rate = 1
chromosome_length =  34 # tinh do dai bit 2^15<( xmax - xmin/sai so cho phep )< 2^16 : chọn 16 bit



if __name__ == "__main__":
    
    data_for_train, data_robot_label, data_fake = create_data()
    print("train_data: ", data_for_train.shape)
    print("Label_data: ", data_robot_label.shape)
    print("Data Fake: ", data_fake.shape)
    X_pre_arr = gen_al(population_size=population_size, num_generations=num_generations, data_fake=data_fake[:10], target_pos=data_robot_label[:10][:, :3],chromosome_length=34, crossover_rate=1, mutation_rate=0.1)





# import torch 
# import torchvision
# from torch.utils.data import DataLoader
# import torch.nn as nn 

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# epoch = 10
# batch_size = 128
# l_rate = 0.001
# train_path  = './dataset/data_robot.npy'
# train_label_path  = './dataset/data_robot_label.npy'

# robot_dataset = RobotDataset(features_path=train_path, labels_path=train_label_path, transform=None)
# robot_dataloader = DataLoader(dataset=robot_dataset, batch_size=batch_size, shuffle=True)


# model = RobotModel(input_size=6, hidden_size=20, out_class=6)
# model = model.to(device=device)
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
# print("training progress")
# history, model = train_model(dataloader=robot_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, metric=calculate_mae, epochs=10)

# print("training done!")


