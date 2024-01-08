import random
import numpy as np 
import os
from utils.forward_kinematics import cal_forward_kinematics

def initialize_population(population_size, chromosome_length):
    population = []
    for _ in range(population_size):
        joint = []
        for _ in range(6):
            gen = ''.join(str(random.randint(0, 1)) for _ in range(chromosome_length))# initialize data: 
            joint.append(gen)
        population.append(joint) 
    return population

def convert_to_float(gen, point):
    float_point = int(point)   + np.sign(point) * int(gen, 2) * (1 / (2 ** len(gen) - 1))
    return  float_point

def fitness_function(X_pre, X_target):
    mse = np.sqrt(np.sum(np.power(X_pre - X_target, 2))) 
    return mse 


def evaluate_population(population, targets=None, theta_arr = None):
    fitness_values = np.zeros(len(population))
    for idx, individual in enumerate(population):
        theta_var  = np.empty(len(individual))
        for i in range(len(individual)):
            theta_i = convert_to_float(individual[i], theta_arr[i])
            theta_var[i] = theta_i
        H = cal_forward_kinematics(theta=theta_var)
        x_pre = np.array([H[0, -1], H[1,-1], H[2,-1]]) 
        # fitness_score = fitness_function(X_pre=x_pre, X_target=x_target)# test when have no testset 
        fitness_score = fitness_function(X_pre=x_pre, X_target=targets)# uncomment when have testset
        fitness_values[idx] = fitness_score
    return fitness_values