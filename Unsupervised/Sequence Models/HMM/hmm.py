#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assigment 11
-------------
HMM to predict the price of Corn

Content
The file composed of simply 2 columns. 
One is the date (weekend) and the other is corn close price.
The period is from 2015-01-04 to 2017-10-01.
The original data is downloaded from Quantopian corn futures price.

Inspiration
William Gann: Time is the most important factor in determining market movements 
and by studying past price records you will be able to prove to yourself history 
does repeat and by knowing the past you can tell the future. 
There is a definite relation between price and time.
"""


# Import libraries
import os
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.cluster import KMeans
from inspect import Signature, Parameter
from itertools import permutations, chain, product

# Paths to data
data_path = os.path.abspath('../data_corn_price')
CORN_2013_2017 = os.path.join(data_path, 'corn2013-2017.txt')
CORN_2015_2017 = os.path.join(data_path, 'corn2015-2017.txt')
OHL = os.path.join(data_path, 'corn_OHLC2013-2017.txt')


# Part I - Data Preprocessing and Exploration
# -------------------------------------------

corn13_17 = pd.read_csv(CORN_2013_2017, names = ("week","price") )
corn15_17 = pd.read_csv(CORN_2015_2017, names = ("week","price"))
OHL_df = pd.read_csv(OHL, names = ("week","open","high","low","close"))

corn13_17.info()
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 248 entries, 0 to 247
#Data columns (total 2 columns):
#week     248 non-null object
#price    248 non-null float64
#dtypes: float64(1), object(1)
#memory usage: 4.0+ KB

'''
The typical structure of a HMM involves a discrete number of latent ("hidden") 
states that are unobserved. The observations, which in our case are corn prices, 
are generated from a state dependent "emission" distribution. 
Emissions are synonymous with observations.

In the discrete HMM case, the emissions are discrete values. 
Conversely, the continuous HMM outputs continuous emissions that are generated 
from the state dependent distribution, which is usually assumed to be Gaussian.

[*] One use case for clustering is to discretize continuous HMM emissions 
    in order to simplify the problem. 
'''

### Generate Clusters --> Quantization
def generate_cluster_assignments(ser, clusters):
    '''
    The function should instantiate a sklearn KMeans class
    with the specified number of clusters and a random_state=24.
    
    The function should return a pandas Series of cluster labels for each
    observation in the sequence.
    
    A KMeans object can be instantiated via:
    clusterer = KMeans(args)
    '''
    quantizier = KMeans(n_clusters=clusters)
    quantizier.fit(ser.values.reshape(-1,1))
    return quantizier.predict(ser.values.reshape(-1,1))


#data_series = pd.Series([1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,6,7,
#                         8,7,6,7,8,6,7,6,7,8,7,7,8,56,57,58,59,57,58,6,7,8,1,2])
#    
#labels = generate_cluster_assignments(data_series, clusters = 3)
##labels --> array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
##                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1])

# Cluster 2013-2017
corn13_17_seq = generate_cluster_assignments(corn13_17[['price']], 5)


# Part II - Components of the HMM
# -------------------------------

'''
A HMM consists of 5 components:

N -- The number of hidden states
A -- State transition matrix
B -- Emission probability matrix
π -- Starting likelihood
(xi,...,xT)(xi,...,xT)  -- Sequence of emissions, or observations
'''




# Part III - Learning the HMM
# ---------------------------

# Cluster 2013-2017
corn13_17_seq = generate_cluster_assignments(corn13_17[['price']], 5)
corn13_17_seq


'''
The Expectation Maximization (EM) algorithm is used to estimate the parameters 
of a HMM given a sequence of observations (aka emissions). 

Here are the general steps for this procedure:

- Initialize a set of parameters for the HMM (π, A, B)

- Conduct the EM algorithm:
    The E Step: 
        Use forward-backward algorithm to calculate the probability of observing 
        the emissions with the given HMM parameters (π, A, B)
    The M Step: 
        Update the HMM parameters so that the sequence of observations are 
        more likely to have come from this particular HMM
    Repeat steps 1 and 2 until the HMM parameters have converged. 
'''


STATE_LIST = ['S1', 'S2']
STATE_TRANS_PROBS = [0.4, 0.6, 0.35, 0.55]

# Almost all functions require this constant as an argument
STATE_LIST = ['S1', 'S2']

# Initialze state transition probabilities (2 states)
STATE_TRANS_PROBS = [0.4, 0.6, 0.35, 0.55]


# Helper Functions
# ----------------

# given a list with unique items, this function will return a new list with all permutations
def make_state_permutations(list_of_unique_states):
    l1 = [''.join(tup) for tup in permutations(list_of_unique_states, 2)]
    l2 = [state+state for state in list_of_unique_states]
    return sorted(l1 + l2)

# helper function in EM function
def _grab_highest_prob_and_state(state_permutations_lst, prob_arr):
    return (prob_arr[np.argmax(prob_arr)], state_permutations_lst[np.argmax(prob_arr)])

# convert dictionary to tuple
def dict_to_tuples(list_of_unique_states, d):
    """
    list_of_unique_states: List of unique state names, as strings
    d: Dictionary of state transition probabilities
    
    
    EXAMPLE:
    s_perms = ['S1S1', 'S1S2', 'S2S1', 'S2S2']
    p_list = [0.1, 0.9, 0.4, 0.6]
    d = {'S1S1': 0.1, 'S1S2': 0.9, 'S2S1': 0.4, 'S2S2': 0.6}
    
    print(dict_to_tuples(d))
    
    OUTPUT:
    {S1: (0.1, 0.9), S2: (0.4, 0.6)}
    """
    
    # Defensive programming to ensure output will be correct
    list_of_unique_states = sorted(list_of_unique_states)
    assert make_state_permutations(list_of_unique_states) == list(d.keys()), \
            "Keys of dictionary must match output of `make_state_permutations(list_of_unique_states)`"
    
    lengths = [len(st) for st in list_of_unique_states]
    final_dict = {}
    for idx, st in enumerate(list_of_unique_states):
        tup = []
        for trans_p in d.keys():
            if trans_p[:lengths[idx]] == st:
                tup.append(d[trans_p])
            else:
                continue
        final_dict[st] = tuple(tup)
        
    return final_dict
    
# convert list of observations to tuple
def obs_to_tuples(list_of_unique_states, d, sequence):
    """
    list_of_unique_states: List of unique state names, as strings
    d: Dictionary of obs transition probabilities
    sequence: the observation sequence
    
    
    EXAMPLE:
    STATE_LIST = ['S1', 'S2']
    d = {'S1_0': 0.1,
         'S1_1': 0.3,
         'S1_2': 0.4,
         'S1_3': 0.15,
         'S1_4': 0.05,
         'S2_0': 0.15,
         'S2_1': 0.2,
         'S2_2': 0.3,
         'S2_3': 0.05,
         'S2_4': 0.3}
    corn15_17_seq = generate_cluster_assignments(corn15_17[['price']], 5)
    
    print(obs_to_tuples(STATE_LIST, d))
    
    OUTPUT:
    {'S1': (0.1, 0.3, 0.4, 0.15, 0.05), 'S2': (0.15, 0.2, 0.3, 0.05, 0.3)}
    """
    
    # Defensive programming to ensure output will be correct
    list_of_unique_states = sorted(list_of_unique_states)
    num_unique_obs = len(np.unique(sequence))
    
    lengths = [len(st) for st in list_of_unique_states]
    final_dict = {}
    for idx, st in enumerate(list_of_unique_states):
        tup = []
        for e_trans in d.keys():
            if e_trans[:lengths[idx]] == st:
                tup.append(d[e_trans])
            else:
                continue
        final_dict[st] = tuple(tup)
        
    return final_dict



# Generate initial probabilities for π, A, B
# ------------------------------------------
def generate_state_trans_dict(list_of_unique_states, **kwargs):
    '''
    
    'list_of_unique_states': list of states as strings
    ''**kwargs': keyword being the state and value is tuple of state transitions.
    <Must be listed in same order as listed in 'list_of_unique_states'>
    
    If **kwargs omitted, transitions are given uniform distribution based on
    number of states.
    
    EXAMPLE1:
    state_params = generate_state_trans_dict(['S1', 'S2', 'S3'])
    
    OUTPUT1:
    {'S1S1': 0.5, 'S2S2': 0.5, 'S1S2': 0.5, 'S2S1': 0.5}
     
    EXAMPLE2:
    state_params = generate_state_trans_dict(['S1', 'S2'], S1=(0.1, 0.9), S2=(0.4, 0.6))
    
    OUTPUT2:
    {'S1S1': 0.1, 'S1S2': 0.9, 'S2S1': 0.4, 'S2S2': 0.6}
    
    '''
    # number of states
    N = len(list_of_unique_states)
    
    # this runs if specific transitions are provided
    if kwargs:
        state_perms = [''.join(tup) for tup in permutations(list(kwargs.keys()), 2)]
        all_permutations = [state+state for state in list_of_unique_states] + state_perms
        pbs = chain.from_iterable(kwargs.values())
        state_trans_dict = {perm:p for perm, p in zip(sorted(all_permutations), pbs)}
        return state_trans_dict
    
    state_perms = [''.join(tup) for tup in permutations(list_of_unique_states, 2)]
    all_permutations = [state+state for state in list_of_unique_states] + state_perms
    state_trans_dict = {perm: (1/N) for perm in all_permutations}
    return state_trans_dict

def generate_emission_prob_dist(list_of_unique_states, sequence, **kwargs):
    '''
    list_of_unique_states: list of states as strings
    sequence: array of observations
    
    EXAMPLE1:
    corn15_17_seq = generate_cluster_assignments(corn15_17[['price']])
    STATE_LIST = ['S1', 'S2']
    
    generate_emission_prob_dist(STATE_LIST, corn15_17_seq, S1=(0.1, 0.3, 0.4, 0.15, 0.05))
    
    OUTPUT1:
    {'S1_0': 0.1,
     'S1_1': 0.3,
     'S1_2': 0.4,
     'S1_3': 0.05,
     'S1_4': 0.05,
     'S2_0': 0.2,
     'S2_1': 0.2,
     'S2_2': 0.2,
     'S2_3': 0.2,
     'S2_4': 0.2}
    '''
    # number of unique obs
    B = list(np.unique(sequence).astype(str))
    
    # this runs if specific transitions are provided
    if kwargs:
        for t in kwargs.values():
            assert len(t) == len(B), "Must provide all probabilities for unique emissions in given state."
            assert round(np.sum(t)) == 1.0, "Given emission probabilities for a state must add up to 1.0"
        for k in kwargs.keys():
            assert k in list_of_unique_states, "Keyword arguments must match a value included in `list_of_unique_states`"
        diff = list(set(list_of_unique_states).difference(kwargs.keys()))
        
        pbs = chain.from_iterable(kwargs.values())
        obs_perms = [state + '_' + str(obs) for state in kwargs.keys() for obs in B]
        
        obs_trans_dict = {perm:p for perm, p in zip(sorted(obs_perms), pbs)}
        
        if diff:
            obs_perms_diff = [state + '_' + obs for state in diff for obs in B]
            obs_trans_dict.update({perm: (1/len(B)) for perm in obs_perms_diff})
            
        return obs_trans_dict
    
    obs_perms = [state + '_' + obs for state in list_of_unique_states for obs in B]
    obs_trans_dict = {perm: (1/len(B)) for perm in obs_perms}
    return obs_trans_dict


def generate_init_prob_dist(list_of_unique_states, **kwargs):
    """
    Examples:
    STATE_LIST = ['S0','S1','S2','S3','S4']
    initial_states = {'S1':.2, 'S2':.3, 'S3':.05, 'S4':.25, 'S0':.2}
    
    print(generate_init_prob_dist(STATE_LIST))
    # --> {'S0': 0.2, 'S1': 0.2, 'S2': 0.2, 'S3': 0.2, 'S4': 0.2}
    
    print(generate_init_prob_dist(STATE_LIST, **initial_states))  ### NOTE: must unpack dictionary with **
    # --> {'S1': 0.2, 'S2': 0.3, 'S3': 0.05, 'S4': 0.25, 'S0': 0.2}
    """
    
    # number of states
    N = len(list_of_unique_states)
    
    # this runs if specific transitions are provided
    if kwargs:
        for t in kwargs.values():
            assert isinstance(t, float), "Must provide probabilities as floats."
            assert t > 0, "Probabilities must be greater than 0."
        assert np.sum(list(kwargs.values())) == 1.0, "Given probabilities must add up to 1.0"
        assert len(kwargs) == len(list_of_unique_states), "Please provide initial probabilities for all states, or leave blank"
        
        # build the prob dictionary
        init_prob_dict = {item[0]: item[1] for item in kwargs.items()}
        return init_prob_dict
    
    init_prob_dist = {state: (1/N) for state in list_of_unique_states}
    return init_prob_dist


# Create State Transition Priors
# ------------------------------
    
# Make permutations of state transition (this len should match len(STATE_TRANS_PROBS))
state_transitions_list = make_state_permutations(STATE_LIST)

# Create transition matrix in form of dictionary
state_transition_probs = {
    trans: prob for trans, prob in zip(state_transitions_list, STATE_TRANS_PROBS)
}
state_transition_probs

# Transform dictionary to be in tuple format
#### - this format is required in the `generate_*` functions used later
A_prior = dict_to_tuples(STATE_LIST, state_transition_probs)
A_prior




