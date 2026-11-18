# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 17:49:04 2025

@author: xy
"""


import os
import sys

my_path = os.getcwd()

os.chdir(my_path)

sys.path.insert(0, my_path)


from math import sqrt
from random import random, sample, randint, choices
# random.sample() is without replacement!!!!
from pandas import DataFrame, Series
from scipy.spatial import distance_matrix
import multiprocessing
import itertools
import numpy as np
from bisect import bisect_left


#import time
#from datetime import timedelta

#start = time.time()

"""GLOBAL PARAMETERS"""
num_processors_to_use = 32 # number of processors to use when multiprocessing
N_tutors_max = 4 # dge effects will show their power if we allow birds that are not on the edges to have more tutors than those on the edges
pop_size = 100
countyears = 50

#general variables and parameters
prop_dev = 0.5
num_of_syll=2000

age = [0,1,2,3,4,5,6,7]
prob_surviving=[0.5, 1 , 0.5, 0.5, 0.5, 0.5, 0.5, 0]
syllables = [syll for syll in range(num_of_syll)] #the x syllables used by the birds
num_cycle = 8000
ind_prod_vec_len = 15 #individual production vector length
young_rep_len = 1 #young individuals' functional repertoire length
learning_occasions = 20
scanty_rep_len = 1 # number of syllable types gotten from "parents" so that young individuals who don't have neighbours would also have a somewhat scanty repertoire


def get_habitat_size(N, x_y_prop, dens):
    habitat_size = N/dens
    habitat_size = round(habitat_size ** (0.5)) ** 2
    if x_y_prop == "equal":
        numof_x_coords = round(sqrt(habitat_size))
    elif x_y_prop == "elongated":
        numof_x_coords = round(sqrt(habitat_size)/2)
    elif x_y_prop == "elongated2":
        numof_x_coords = round(habitat_size/3)
    elif x_y_prop == "one-line":
        numof_x_coords = 1 # defining it as a float as in the other two cases it will also be a float type
    numof_y_coords = round(habitat_size/numof_x_coords)
    return numof_x_coords, numof_y_coords



def create_habitat_coord_pairs(num_of_x_coords, num_of_y_coords):
    coord_pairs = [[x, y] for x in range(num_of_x_coords) for y in range(num_of_y_coords)]
    return coord_pairs



def my_mean(lst):
    return sum([i for i in lst])/len(lst)



# function for weighted sampling without replacement:
def wsample(population, weights, k=1):
    accum = list(itertools.accumulate(weights))
    total = accum[-1]
    sampl = {}
    while len(sampl) < k:
        index = bisect_left(accum, total * random()) # the larger the original weight the higher the chance that the value of total * random() will fall between the given value and its lower neighbour
        sampl[index] = population[index]
    return list(sampl.values())



def learning(individual, tutors, intensity, learning_mistake):
    indi = num_of_syll
    #collecting the syllables used by  the juveniles' tutors
    Px_list = []
    tutorsrep_flat = [syllable for current_tutor in tutors for syllable in birds_syll_rep[current_tutor]]
    set_tutorsrep = list(set(tutorsrep_flat))
    Fx = [tutorsrep_flat.count(current_syllable) for current_syllable in set_tutorsrep]
    Px_list = [(Fx[i]**intensity) for i in range(len(Fx))]        
    indi = choices(set_tutorsrep, weights = Px_list, k=1)[0]
    pm = random()
    if pm <= learning_mistake: # a mistake in learning happens
        #print("A mistake is being made, turning", indi, "into:")
        indi = sample(range(num_of_syll), 1)[0]
        #print(indi)
    # changing a RANDOM syllable in the ind's PV to the new, learned syllable
    birds_syll_rep[individual][randint(0, ind_prod_vec_len-1)] = indi

        

def choose_tutors(population, choosing_prob_matrix, individual, N_tutors): # function for choosing tutors based on the already created matrix of choosing probabilities:
    choosing_prob = np.concatenate((choosing_prob_matrix[individual][:individual], choosing_prob_matrix[individual][individual+1:])) # excluding the weight which belongs to the given individual
    N_neighbours = sum(choosing_prob > 0)
    if N_neighbours < N_tutors: # if there are less neighbours within audibility distance than the potential number of tutors
        tutors = wsample(population[:individual]+population[individual+1:], k = N_neighbours, weights = choosing_prob) # sampling the population except for the learning individual itself based on the weights
    else:
        tutors = wsample(population[:individual]+population[individual+1:], k = N_tutors, weights = choosing_prob) # sampling the population except for the learning individual itself based on the weights
    return tutors


def get_birds_syll_rep(): # function for getting the birds_syll_rep during the simulation; NOTE: the initial birds_syll_rep is assigned differently
    birds_syll_rep2 = [birds_syll_rep[i] if i in survivors else choices(new_pop_syll_rep_freq["syll_types"], weights = new_pop_syll_rep_freq['count'], k=ind_prod_vec_len) if random()<=migr else sample([syllable for current_tutor in birds_tutors[i] for syllable in birds_syll_rep[current_tutor]], young_rep_len*round(ind_prod_vec_len/young_rep_len)) if len(birds_tutors[i])>0 else sample(new_pop_syll_rep, scanty_rep_len)*round(ind_prod_vec_len/scanty_rep_len)  for i in population]
    return birds_syll_rep2


def song_learning(a, migr, lm, density, audibility, x_y_rate, num_param_combs_repeat, habitat_x, habitat_y):
    #
    # print(pop_size, density, x_y_rate, habitat_x, habitat_y)
    global population, birds_syll_rep, survivors, birds_tutors, new_pop_syll_rep, new_pop_syll_rep_freq
    if x_y_rate == "elongated2":
        global pop_size
        pop_size = round(habitat_x*habitat_y*density)
    population = list(range(pop_size))
    
    # the habitat:
    habitat_coord_pairs = create_habitat_coord_pairs(habitat_x, habitat_y)
    # attributes of birds:
    birds_ages = [0 for x in population]
    birds_prob_surviving = [prob_surviving[i] for i in birds_ages]
    birds_syll_rep = [sample(syllables, young_rep_len)*round(ind_prod_vec_len/young_rep_len) for i in population] # this way each individual repertoire will be a list in a list
    birds_coord_pairs = sample(habitat_coord_pairs, pop_size)
        
    pop_indrep = []
    pop_indrep_colnames = []
    pop_indcoords = []
    pop_repsize_through_years = []
    pop_repsize_through_years_colnames = []

    birds_distance_matrix = distance_matrix(birds_coord_pairs, birds_coord_pairs)
    # birds_distance_matrix[birds_distance_matrix > audibility] = 1000
    choosing_prob_matrix = 1/(birds_distance_matrix + 1) # array of possibilities of choosing a given individual as a tutor calculated as p_AB = 1/(euclidean_dist_AB+1)
    choosing_prob_matrix[birds_distance_matrix > audibility] = 0
    birds_tutors = [choose_tutors(population, choosing_prob_matrix, i, N_tutors = N_tutors_max) for i in population]
    #
    for c in range(num_cycle):
        for lo in range(learning_occasions):
            for i in population:
                if birds_ages[i]==0:
                    tutors = birds_tutors[i]
                    if len(tutors)>=1: # with low density values it could happen that the learning individual doesn't have neighbours so in that case learning won't happen
                        learning(individual = i, tutors = tutors, intensity = a, learning_mistake = lm)
        
        new_pop_syll_rep = [current_syllable for individual in population for current_syllable in birds_syll_rep[individual]]

        if (c+1) % countyears == 0 or c >= (num_cycle-16):
            sum_use = []
            #
            for i in range(len(syllables)):
                sum_use.append(sum([1 for x in new_pop_syll_rep if x == i]))
            #
            sum_sum_use = sum(sum_use)
            pop_repsize_through_years.append([x/sum_sum_use for x in sum_use])
            pop_repsize_through_years_colnames.append(str(c))
        #
        #
        #
        # creating a dataframe that will be built based on the current repertoire of the population but it will contain different syllable types; We will use this later, to fill up the immigrants' repertoire.
        new_pop_syll_rep_freq = DataFrame(Series(new_pop_syll_rep).value_counts()).rename_axis("syll_types").reset_index()
        indices_to_replace = sample(range(new_pop_syll_rep_freq.shape[0]), round(new_pop_syll_rep_freq.shape[0]*prop_dev))
        sylls_to_replace_with = sample(syllables, round(new_pop_syll_rep_freq.shape[0]*prop_dev)) # this is needed as otherwise there would be a chance that one syllable would b selected more than once which would influence the relative frequncies of syllables of immigrants
        for ind in indices_to_replace:
            new_pop_syll_rep_freq["syll_types"][ind] = sylls_to_replace_with.pop()
        # #
        if c >= (num_cycle-10):
            pop_indrep_colnames.extend([str(c)+'_'+str(i) for i in population])
            pop_indrep.extend([birds_syll_rep[i] for i in population])
            pop_indcoords.extend([birds_coord_pairs[i] for i in population])
        #next step: some of them gotta die
        survivors = [i for i in population if random() < birds_prob_surviving[i]]
        #updating features of birds:
        birds_ages = [birds_ages[i]+1 if i in survivors else 0 for i in population]
        # resetting the habitat and birds' coordinates:
        free_coord_pairs = [i for i in habitat_coord_pairs if i not in birds_coord_pairs] + [birds_coord_pairs[i] for i in population if i not in survivors]
        set_new_birds_coord_pairs = sample(free_coord_pairs, pop_size - len(survivors))
        birds_coord_pairs = [birds_coord_pairs[i] if i in survivors else set_new_birds_coord_pairs.pop() for i in population]
        # redefining tutors (=neighbours):
        birds_distance_matrix = distance_matrix(birds_coord_pairs, birds_coord_pairs)
        choosing_prob_matrix = 1/(birds_distance_matrix + 1) # array of possibilities of choosing a given individual as a tutor calculated as p_AB = 1/(euclidean_dist_AB+1)
        choosing_prob_matrix[birds_distance_matrix > audibility] = 0
        birds_tutors = [choose_tutors(population, choosing_prob_matrix, i, N_tutors = N_tutors_max) for i in population]
        # and then assigning the repertoires of new birds:
        birds_syll_rep = get_birds_syll_rep()

    #
    pop_repsize_through_years_df = DataFrame(pop_repsize_through_years).transpose()
    pop_repsize_through_years_df.columns = pop_repsize_through_years_colnames
    pop_repsize_through_years_df.to_csv('pop10rep_' + str(num_cycle) + 'x' + str(learning_occasions) + 'sys' + str(num_of_syll) + '_density' + str(density) + '_Nt' + str(N_tutors_max) + '_popsize' + str(pop_size) + '_xyrate_' + str(x_y_rate) + '_audibility' + str(audibility) + '_' + str(num_param_combs_repeat) + '.csv', encoding = "utf-8")
    pop_indrep_df = DataFrame(pop_indrep).transpose()
    pop_indrep_df.columns = pop_indrep_colnames
    pop_indrep_df.to_csv('pop_indrep_'+str(num_cycle)+'x'+str(learning_occasions)+'sys'+str(num_of_syll) + '_density' + str(density) + '_Nt' + str(N_tutors_max) + '_popsize' + str(pop_size) + '_xyrate_' + str(x_y_rate) + '_audibility' + str(audibility) + '_' + str(num_param_combs_repeat) + '.csv', encoding = "utf-8")
    pop_indcoords_df = DataFrame(pop_indcoords).transpose()
    pop_indcoords_df.columns = pop_indrep_colnames # it has the same colnames as pop_indrep_df
    pop_indcoords_df.to_csv('pop_indcoords_'+str(num_cycle)+'x'+str(learning_occasions)+'sys'+str(num_of_syll) + '_density' + str(density) + '_Nt' + str(N_tutors_max) + '_popsize' + str(pop_size) + '_xyrate_' + str(x_y_rate) + '_audibility' + str(audibility) + '_' + str(num_param_combs_repeat) + '.csv', encoding = "utf-8")



"""ADJUSTABLE PARAMETERS"""
alpha = 1.1

migr = 0.03

lm = 0.1


# population sizes should be all square numbers as it makes it easier to later define the size of the space
density1 = 1
density2 = 0.5
density3 = 0.25

x_y_rate0 = "equal" # when x and y are equal e.g. the space is a square
x_y_rate_high = "elongated" # when space is a bit elongated
x_y_rate_high2= "elongated2"
x_y_rate_high3 = "one-line" # when space is basically one line

audibility1 = 1
audibility2 = 2
audibility3 = 3
audibility4 = 100

num_param_combs_repeat = range(10)

params = {
        'alpha' : [alpha],
        'migr' : [migr],
        'lm' : [lm],
        'density' : [density1, density2, density3],
        'x_y_rate' : [x_y_rate0, x_y_rate_high, x_y_rate_high2, x_y_rate_high3],
        'audibility' : [audibility1, audibility2, audibility3, audibility4],
        'num_param_combs_repeat' : [x for x in num_param_combs_repeat]
        }

num_param_comb = len(params['alpha']) * len(params['migr']) * len(params['lm']) * len(params['density']) * len(params['x_y_rate']) * len(params['audibility']) * len(num_param_combs_repeat)


keys = list(params)
param_combs = DataFrame(columns = ['alpha', 'migr', 'lm', 'density', 'x_y_rate', 'audibility', 'num_param_combs_repeat'], index = [x for x in range(num_param_comb)])
rownum = 0

for values in itertools.product(*map(params.get, keys)):
    param_combs.loc[rownum] = list(values)
    rownum = rownum+1

# condition on which parameter values to test (optional):
 #   param_combs[(param_combs['density'] == 0.25) & (param_combs['x_y_rate'] == 'equal') & (param_combs['audibility'] == 1)]


habitat_sizes = param_combs.apply(lambda x : get_habitat_size(N = pop_size, x_y_prop = x['x_y_rate'], dens = x['density']), axis = 1)

param_combs['habitat_size_x'] = list(zip(*habitat_sizes))[0]
param_combs['habitat_size_y'] = list(zip(*habitat_sizes))[1]


def main():
    with multiprocessing.Pool(num_processors_to_use) as pool:
        pool.starmap(song_learning, zip(param_combs.loc[:,'alpha'], 
                                        param_combs.loc[:,'migr'], 
                                        param_combs.loc[:,'lm'], 
                                        param_combs.loc[:,'density'],
                                        param_combs.loc[:,'audibility'],
                                        param_combs.loc[:, 'x_y_rate'],
                                        param_combs.loc[:,'num_param_combs_repeat'],
                                        param_combs.loc[:, 'habitat_size_x'],
                                        param_combs.loc[:, 'habitat_size_y']))
                
        
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
