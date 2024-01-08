
import os 
import numpy  as np
import pandas as pd
import time
import random 
from .initialization import initialize_population, evaluate_population, fitness_function, convert_to_float
from .functions import crossover, roulette_wheel_selection, mutate
from utils.forward_kinematics import cal_forward_kinematics


def gen_al(population_size = 100, num_generations = 100, data_fake=None, target_pos=None, chromosome_length=34, crossover_rate=0.1, mutation_rate = 0.01):
    x_pre_arr = np.zeros_like(target_pos)
    for i in range(len(data_fake)):
        print(f"Point: {i} | Target POS: {target_pos[i]} | Fake theta: {data_fake[i]} \n")
        population = initialize_population(population_size, chromosome_length)# khởi tạo môi trường 
        best_individual = None
        best_fitness = float('inf')
        start = time.time() 
        
        for generation in range(num_generations):
            start_gen = time.time()
            fitness_values = evaluate_population(population=population,targets=target_pos[i], theta_arr=data_fake[i]) # tính fitness score trên toàn bộ population 
            mating_pool = roulette_wheel_selection(population, fitness_values, population_size=population_size)
            new_population = []
            j = 0
            
            # Tính toán để lấy cá thể tốt nhất
            min_fitness = min(fitness_values)
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_individual = population[np.argmin(fitness_values)].copy()
            else: 
                best_individual = population[np.argmin(fitness_values)].copy()

            while j <= population_size:
                parent1, parent2 = random.choices(mating_pool, k=2)
                new_population.extend([parent1, parent2])
                if random.random() < crossover_rate:
                    offspring1, offspring2 = crossover(parent1, parent2)
                    offspring1 = mutate(offspring1, mutation_rate= mutation_rate)
                    offspring2 = mutate(offspring2, mutation_rate=mutation_rate)
                    new_population.extend([offspring1, offspring2])
                else:
                    offspring1, offspring2 = parent1, parent2
                j +=1 
            # thêm cá thể tốt nhất từ lần generation trước vào popuplation mới
            if best_individual is not None:
                new_population.append(best_individual)

            population = np.array(new_population)
            
            # Find the best individual from the final population
            end_fitness_values = evaluate_population(population=population, targets=target_pos[i], theta_arr=data_fake[i])
            if best_individual is None or min(end_fitness_values) < best_fitness:
                best_individual = population[np.argmin(end_fitness_values)]
                theta_var = np.zeros(len(best_individual))
                for j in range(len(best_individual)):
                    theta_j = convert_to_float(best_individual[j], data_fake[i][j])
                    theta_var[j] = theta_j
                H = cal_forward_kinematics(theta=theta_var)
                x_pre = np.array([H[0, -1], H[1,-1], H[2,-1]])
            else:
                best_individual = population[np.argmin(end_fitness_values)]
                theta_var = np.zeros(len(best_individual))
                for j in range(len(best_individual)):
                    theta_j = convert_to_float(best_individual[j], data_fake[i][j])
                    theta_var[j] = theta_j
                H = cal_forward_kinematics(theta=theta_var)
                x_pre = np.array([H[0, -1], H[1,-1], H[2,-1]])
            best_fitness = min(end_fitness_values)
            end_gen = time.time() - start_gen
            print(f"Generation: {generation} | x = {x_pre} | x_tar  = {target_pos[i]}| best_fitness_value(MSE) = {best_fitness}| Time Loop: {end_gen} \n")
            
        x_pre_arr[i] = x_pre
    return x_pre_arr   
            
            
            
