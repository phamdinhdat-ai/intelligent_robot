import numpy as np 
import matplotlib.pyplot as plt 
from numpy import sin, cos


def get_transform_matrix(theta, d, a, alpha):
    
    T = np.array([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a*cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0, sin(alpha), cos(alpha), d], 
        [0, 0, 0, 1]
    ])
    return T


def cal_forward_kinematics(theta):
    T_0 = np.array([[1, 0,  0,  0], [0,  1,  0,  0],  [0,  0,  1,  0], [0, 0, 0, 1]])
    DH_table = np.array([
        [np.deg2rad(theta[0]), np.deg2rad(theta[1]), 0, np.deg2rad(theta[3]), np.deg2rad(theta[4]), np.deg2rad(theta[5])],
        [0, 120, theta[2], 0, 0 , 0],
        [0, 0 , 0 , 0 , 0 , 0 ],
        [-90, 0, 90, -90, 90, 0]   
    ])
    for i in range(len(theta)):
        T = get_transform_matrix(theta=DH_table[0][i], d=DH_table[1][i], a=DH_table[2][i], alpha=DH_table[3][i])
        T_0 = T_0.dot(T)
    return T_0
        
    
     