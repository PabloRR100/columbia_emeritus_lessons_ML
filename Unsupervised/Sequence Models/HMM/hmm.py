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


# Create Emission Priors
# ----------------------
# Manually initialize emission probabilities - in format 1
B_format1 = {
    'S1_0': 0.1,
    'S1_1': 0.3,
    'S1_2': 0.4,
    'S1_3': 0.15,
    'S1_4': 0.05,
    'S2_0': 0.15,
    'S2_1': 0.2,
    'S2_2': 0.3,
    'S2_3': 0.05,
    'S2_4': 0.3
}

# Convert emission matrix to format 2
B_format2 = obs_to_tuples(STATE_LIST, B_format1, corn13_17_seq)
B_format2

B_prior = obs_to_tuples(STATE_LIST, B_format1, corn13_17_seq)


# Create initial state probability π
# ----------------------------------

pi__init = {'S1': 0.4 , 'S2': 0.6}
pi = generate_init_prob_dist(STATE_LIST, **pi__init)
pi



# EXPECTATION MAXIMIZATION ALGORITHM
# -----------------------------------

'''
Finally, we need to create a data structure that will hold all of our 
probability calculations until we are finished computing E Step

For this task, we will take advantage of a powerful data structure 
from the `collections` module: `namedtuple`.
'''

def generate_obs_data_structure(sequence):
    #  sequence: 1D numpy array of observations
    ObservationData = namedtuple(
        'ObservationData',
        ['prob_lst', 'highest_prob', 'highest_state']
    )
    return {index+1: ObservationData for index in np.arange(len(sequence)-1)}


# STEP 1 -> Estimate Probabilities
'''
This step involves using the Forward-Backward algorithm to calculate the 
probability of observing a sequence, given a set of HMM parameters
'''

# Enforce an Argument Signature on following function to prevent errors with **kwargs
params = [Parameter('list_of_unique_states', Parameter.POSITIONAL_OR_KEYWORD),
         Parameter('sequence', Parameter.POSITIONAL_OR_KEYWORD),
         Parameter('A', Parameter.KEYWORD_ONLY, default=generate_state_trans_dict),
         Parameter('B', Parameter.KEYWORD_ONLY, default=generate_emission_prob_dist),
         Parameter('pi', Parameter.KEYWORD_ONLY, default=generate_init_prob_dist)]

sig = Signature(params)

def calculate_probabilities(list_of_unique_states, sequence, **kwargs):
    
    # enforce signature to ensure variable names
    bound_values = sig.bind(list_of_unique_states, sequence, **kwargs)
    bound_values.apply_defaults()

    
    # grab params that are left to default values
    param_defaults = [(name, val) for name, val in bound_values.arguments.items() if callable(val)]
    
    # grab non-default params
    set_params = [(name, val) for name, val in bound_values.arguments.items() if isinstance(val, dict)]
    
    # this will run if any default hmm parameters are used
    if param_defaults:
        for name, val in param_defaults:
            if name == 'B':
                B = val(list_of_unique_states, sequence)
            elif name == 'A':
                A = val(list_of_unique_states)
            elif name == 'pi':
                pi = val(list_of_unique_states)
            else:
                continue
    
    # this will run if kwargs are provided        
    if set_params:
        for name, val in set_params:
            if name == 'B':
                B = generate_emission_prob_dist(list_of_unique_states, sequence, **val)
            elif name == 'A':
                A = generate_state_trans_dict(list_of_unique_states, **val)
            elif name == 'pi':
                pi = generate_init_prob_dist(list_of_unique_states, **val)
            else:
                continue
                
    # instantiate the data structure
    obs_probs = generate_obs_data_structure(sequence)

    # all state transitions
    state_perms = make_state_permutations(list_of_unique_states)

    # for every transition from one observation to the next, calculate probability of going from Si to Sj
    # loop through observations
    for idx, obs in enumerate(sequence):

        if idx != 0:  # check if this is the first observation
            # instantiate the namedtuple for this observation
            obs_probs[idx] = obs_probs[idx]([], [], [])

            # loop through each possible state transition
            for st in state_perms:
                
                # calculate prob of current obs for this state
                prev_prob = pi[st[:2]] * B[st[:2]+'_'+str(sequence[idx-1])]

                # calculate prob of previous obs for this state
                curr_prob = A[st] * B[st[2:]+'_'+str(obs)]

                # combine these two probabilities
                combined_prob = round(curr_prob * prev_prob, 4)

                # append probability to the list in namedtuple
                obs_probs[idx].prob_lst.append(combined_prob)

            # check for highest prob of observing that sequence
            prob_and_state = _grab_highest_prob_and_state(state_perms, obs_probs[idx].prob_lst)
            obs_probs[idx].highest_prob.append(prob_and_state[0])
            obs_probs[idx].highest_state.append(prob_and_state[1])

        else: # this is the first observation, exit loop.
            continue
    return (obs_probs, A, B, pi)


ob_prob, A, B, pi = calculate_probabilities(STATE_LIST, corn13_17_seq, A=A_prior, B=B_prior, pi=pi)


# STEP 2 -> Update Parameters

# Update A: States transitions
# This function sums all of the probabilities and 
# outputs a new (un-normalized) state transition matrix
def new_state_trans(STATE_LIST, probabilities):
    state_perms = make_state_permutations(STATE_LIST)
    sums_of_st_trans_prob = {p:0 for p in state_perms}
    highest_prob_sum = 0
    for obs in probabilities:
        highest_prob_sum += probabilities[obs].highest_prob[0]
        for i, p in enumerate(sums_of_st_trans_prob):
            sums_of_st_trans_prob[p] += probabilities[obs].prob_lst[i]
    
    for key in sums_of_st_trans_prob:
        sums_of_st_trans_prob[key] = sums_of_st_trans_prob[key] / highest_prob_sum
    
    # finally, normalize so the rows add up to 1
    for s in STATE_LIST:
        l = []
        for k in sums_of_st_trans_prob:
            if s == k[:2]:
                l.append(sums_of_st_trans_prob[k])
        for k in sums_of_st_trans_prob:
            if s == k[:2]:
                sums_of_st_trans_prob[k] = sums_of_st_trans_prob[k] / sum(l)
    
    return sums_of_st_trans_prob

# Update and normalize posterior state transition
A_posterior = new_state_trans(STATE_LIST, ob_prob)
A_posterior

# Convert state transition to "format 2" so it can be
# used as input in the next iteration of "E" step
A_posterior = dict_to_tuples(STATE_LIST, A_posterior)


# Update B: emissions probabilities
##### tally up all observed sequences
def observed_pairs(sequence):
    observed_pairs = []
    for idx in range(len(sequence)-1):
        observed_pairs.append((sequence[idx], sequence[idx+1]))
    return observed_pairs

def make_emission_permutations(sequence):
    unique_e = np.unique(sequence)
    return list(product(unique_e, repeat = 2))

make_emission_permutations([1,1,0, 2])
make_emission_permutations([0,1,0,3,0])

def find_highest_with_state_obs(prob_pairs, state, obs):
    for pp in prob_pairs:
        if pp[0].count((state,obs))>0:
            return pp[1]

def normalize_emissions(b_tuple_format):
    new_b_dict = {}
    for key, val in b_tuple_format.items():
        denominator = sum(val)
        new_lst = [v/denominator for v in val]
        new_b_dict[key] = tuple(new_lst)
    return new_b_dict


# we are ready to update the emission probabilities with the function below
def emission_matrix_update(sequence, state_list, A, B, pi):
    state_pairs = list(product(state_list, repeat = 2))
    obs_pairs = observed_pairs(sequence)
    
    new_B = {}
    for obs in np.unique(sequence): # For every unique emission
        
        # Find all the sequence-pairs that include that emission
        inc_seq = [seq for seq in obs_pairs if seq.count(obs)>0]

        # Collector for highest-probabilities
        highest_pairs = []
        
        # For each sequence-pair that include that emission
        for seq in inc_seq:

            prob_pairs = []
            
            # Go through each potential pair of states
            for state_pair in state_pairs:
                
                state1, state2 = state_pair
                obs1, obs2 = seq
                
                # Match each state with it's emission
                assoc_tuples = [(state1, obs1),
                                (state2, obs2)]
                
                # Calculate the probability of the sequence from state
                prob = pi[state1] * B[state1+"_"+str(obs1)]
                prob *= A[state1+state2]*B[state2+"_"+str(obs2)]
                prob = round(prob,5)
                # Append the state emission tuples and probability
                prob_pairs.append([assoc_tuples, prob])
    
            # Sort probabilities by maximum probability
            prob_pairs = sorted(prob_pairs, key = lambda x: x[1], reverse = True)
            
            # Save the highest probability
            to_add = {'highest':prob_pairs[0][1]}
            # Find the highest probability where each state is associated
            # With the current emission
            for state in STATE_LIST:
                
                highest_of_state = 0
                
                # Go through sorted list, find first (state,observation) tuple
                # save associated probability

                for pp in prob_pairs:
                    if pp[0].count((state,obs))>0:
                        highest_of_state = pp[1]
                        break
                        
                to_add[state] = highest_of_state
            
            # Save completed dictionary
            highest_pairs.append(to_add)
        
        # Total highest_probability
        highest_probability =sum([d['highest'] for d in highest_pairs])
        
        # Total highest probabilities for each state; divide by highest prob
        # Add to new emission matrix
        for state in STATE_LIST:
            new_B[state+"_"+str(obs)]= sum([d[state] for d in highest_pairs])/highest_probability
            
        
    return new_B

nb = emission_matrix_update(corn13_17_seq,STATE_LIST, A,B,pi)
nb

#The emission probabilities are updated, but they need to be normalized. 
#To do this, we will convert to dictionary to the `key: tuple` format and 
#normalize so that the probabilities add up to 1.

B_ = obs_to_tuples(STATE_LIST, nb, corn13_17_seq)
B_posterior = normalize_emissions(B_)

# normalized state transition posterior:
A_posterior

# normalized emission posterior probabilities
B_posterior


# STEP 3 -> Repeat until parameters convergence
ob_prob2, A2, B2, pi2 = calculate_probabilities(STATE_LIST, corn13_17_seq, A=A_posterior, B=B_posterior, pi=pi)
ob_prob2

A_post2 = new_state_trans(STATE_LIST, ob_prob2)  # update and normalize state transition matrix again
A_post2 = dict_to_tuples(STATE_LIST, A2)  # convert to `key: tuple` format
A_post2

# update emissions matrix again
nb2 = emission_matrix_update(corn13_17_seq, STATE_LIST, A2, B2, pi)  # update emissions matrix again
B_post2 = obs_to_tuples(STATE_LIST, nb2, corn13_17_seq)  # convert emission posterior to `key:tuples` format
B_post2 = normalize_emissions(B_post2)  # normalize emissions probabilities
B_post2


