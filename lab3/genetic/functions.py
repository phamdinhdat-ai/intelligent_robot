import numpy as np 
import random 


def crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        crossover_point1 = random.choice(range(0, 17))
        crossover_point2 = random.choice(range(17, 34))
        gen1 = parent1[i][:crossover_point1] + parent2[i][crossover_point1:crossover_point2] + parent1[i][crossover_point2:]
        gen2 = parent2[i][:crossover_point1] + parent1[i][crossover_point1:crossover_point2] + parent2[i][crossover_point2:]
        child1.append(gen1)
        child2.append(gen2)
    return child1, child2



# Selection - using roulette wheel selection
def roulette_wheel_selection(population, fitness_values, population_size):
    total_fitness = np.sum(fitness_values)
    normalized_fitness = fitness_values / total_fitness # normalize fitness scores 

    wheel = np.cumsum(normalized_fitness)# tính tổng lần lượt theo kết quả fitness scores

    selected = []
    for _ in range(population_size):
        pointer = np.random.random()# chọn random tỉ lệ [0 ; 1 ]
        idx = np.searchsorted(wheel, pointer)#  lấy ra index tại vị trí tổng lần lượt bằng giá trị pointer (r)
        selected.append(population[idx])# lần lượt lấy từ môi trường cá thể có index đó 

    return selected



# Mutation
def mutate(offspring, mutation_rate=0.01):
    offspring_next = []
    for individual in offspring:
        mutated_individual = ''
        for bit in individual:
            if random.random() < mutation_rate:
                mutated_bit = '0' if bit == '1' else '1'  # one point mutation (flip the bit)
                mutated_individual += mutated_bit
            else:
                mutated_individual += bit # neu random value nho hon mutation rate thi se return parents
        offspring_next.append(mutated_individual)
    return offspring_next


