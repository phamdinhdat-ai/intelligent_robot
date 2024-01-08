import numpy as np 
from .forward_kinematics import get_transform_matrix, cal_forward_kinematics
from .cal_roll_pitch_yaw import cal_orientation



    
    
def create_data():
    theta_1 = np.expand_dims(np.linspace(start=-100, stop=100, num=1000), axis = 1)
    theta_2 = np.expand_dims(np.linspace(start=-100, stop=100, num=1000), axis = 1)
    theta_3 = np.expand_dims(np.zeros(1000), axis = 1)
    theta_4 = np.expand_dims(np.linspace(start=-100, stop=100, num=1000),axis = 1) 
    theta_5 = np.expand_dims(np.linspace(start=-100, stop=100, num=1000), axis = 1)
    theta_6 = np.expand_dims(np.linspace(start=-100, stop=100, num=1000), axis = 1)
    d_3 = np.expand_dims(np.linspace(start=10, stop=120,num=1000), axis = 1)
    data_for_train  = np.concatenate([theta_1, theta_2, d_3, theta_4, theta_5, theta_6], axis=1)
    data_label = []
    for i in range(len(data_for_train)):
        H = cal_forward_kinematics(theta=data_for_train[i])
        # print(H)
        ox, oy, oz = cal_orientation(H=H)
        x, y , z = H[0, -1], H[1,-1], H[2,-1]
        # print(f"X: {H[0,-1]} |Y: {H[1,-1]}|Z: {H[2,-1]} |O_x: {ox} | O_y: {oy} |O_z: {oz}")
        data_label.append([x, y, z, ox, oy, oz])
    data_robot_label = np.array(data_label)
    data_fake = data_for_train + np.array([[np.round(np.random.uniform(), decimals=10) for _ in range(6)] for _ in range(1000)])
    return data_for_train, data_robot_label, data_fake
    




