#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:51:05 2019
@author: pabloruizruiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read in Data
df = pd.read_excel("default of credit card clients.xls", header = 1)
df.rename(columns = {"PAY_0":"PAY_1"}, inplace = True) #renaming mis-named column
df.head()

resp = 'default payment next month'

# Q1 - Default Payment
ones = df[resp].sum()
zeros = df.shape[0] - ones
print('Default = 0: {} \n Default = 1: {}'.format(zeros, ones))


# PART II - INFORMATION THEORY

def gini_coef(distribution: list):
    # distribution = [[s11, s12], [s21, s22]]
    gini = 1
    
    b1 = distribution[0]
    b2 = distribution[1]
    tot_1 = sum(b1)
    tot_2 = sum(b2)
    freq_11 = b1[0] / tot_1
    freq_12 = b1[1] / tot_1
    freq_21 = b2[0] / tot_2
    freq_22 = b2[1] / tot_2
    
    gini1 = 1 - (freq_11**2 + freq_12**2)
    gini2 = 1 - (freq_21**2 + freq_22**2)
#    for i in distribution:
#        gini -= (i / np.sum(distribution))**2
    return (gini1 + gini2) / 2

ame = [[175, 330], [220, 120]]
ger = [[110,  60], [285, 390]]

gini_coef(ame)
gini_coef(ger)

