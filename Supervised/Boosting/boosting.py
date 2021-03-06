#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 08:48:41 2019
@author: pabloruizruiz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Simple_Binary(object):
    
    def __init__(self):
        pass
    
    def fit(self, X,y):
        """
            1. Find best split in X
                - According to entropy
            2. After finding split, assign:
                - self.col_idx
                - self.split_value
                - self.left_pred
                - self.right_pred
        """
        pass
    
    def predict(self, X):
        """
            1. Make predictions given values calculated
                in the `.fit(X,y)` method.
            2. return predictions as numpy array.
        """
        pass
    
    
def simple_binary_tree_fit(X,y):
    """
    Positional arguments:
        X -- a numpy array of numeric observations:
            Assume rows are separate observations, columns are features
        y -- a numpy array of binary labels:
            *Assume labels are 1 for "True" and 0 for "False"*
            
    1. Find best split in X
        - According to entropy
    2. After finding split, return:
        - col_idx - index of column used to split data
        - split_value - value upon which data is split
        - left_pred - The prediction for observation <= split_value
        - right_pred - The prediciton for observation > split_value        
    """
    
    # create variable "best_split" which will hold:
    # (col_number, split_value, entropy)
    best_split = (-1,-1,1)
    
    # loop through each column in X, keeping track of the column index.
    # # # Note, taking the transpose of X -- X.T -- yeilds columns in this "for" loop
    for col_idx, col in enumerate(X.T):
        
        # Find potential split values within column using `find_splits(col)`
        splits = find_splits(col) ### <------
        
        # For each split, calculate entropy
        for s in splits:
            ent = ent_from_split(col, s, y) ### <------
            
            # Check if calculated entropy is less than previous "best"
            if ent < best_split[2]:
                best_split = (col_idx, s, ent)
    
    # Now, the "best split" has been found.
    # create "left" and "right" predictions for the best_split
    # The "left" predictions is for when `observation` <= `split_value`
    # The "right" prediction is for when `observation` > `split_value`
    # Each prediction will either be 1 for "True" or 0 for "False"
    
    left_pred, right_pred = pred_from_split(X, y, *best_split[:2]) ### <------
    
    col_idx, split_value = best_split[:2]
    
    # return:
    # - the index of the column to split on.
    # - the value to split that column on
    # - the prediction for rows with observations in that column less than or equal to the split
    # - the prediction for rows with observations in that column greater than the split
    
    return col_idx, split_value, left_pred, right_pred



### GRADED
### Code a function called `find_splits`.
### ACCEPT a 1-dimensional numpy array as input.
### RETURN a numpy.array of "split values"

### "Split values" are the mid-points between the values in the sorted list of unique values.

### e.g., Input of np.array([1, 3, 2, 3, 4, 6])
### Yields a sorted-unique list of: np.array([1, 2, 3, 4, 6])
### Then the "splits" in between those values will be: np.array([1.5, 2.5, 3.5, 5])

### YOUR ANSWER BELOW

def find_splits(col):
    """s
    Calculate and return all possible split values given a column of numeric data
    
    Positional argument:
        col -- a 1-dimensional numpy array, corresponding to a numeric
            predictor variable.
    
    Example:
        col = np.array([0.5, 1. , 3. , 2. , 3. , 3.5, 3.6, 4. , 4.5, 4.7])
        splits  = find_splits(col)
        print(splits) # --> np.array([0.75, 1.5, 2.5, 3.25, 3.55, 3.8, 4.25, 4.6])
        
    """
    unique = np.unique(sorted(col))
    splits = np.array([np.mean([unique[i], unique[i+1]]) for i in range(len(unique)-1)])
    return splits

col = np.array([0.5, 1. , 3. , 2. , 3. , 3.5, 3.6, 4. , 4.5, 4.7])
splits  = find_splits(col)
print(splits) # --> np.array([0.75, 1.5, 2.5, 3.25, 3.55, 3.8, 4.25, 4.6])


def entropy(class1_n, class2_n):
    # If all of one category, log2(0) does not exist,
    # and entropy = 0
    if (class1_n == 0) or (class2_n == 0):
        return 0

    # Find total number of observations 
    total = class1_n + class2_n

    # find proportion of both classes
    class1_proprtion = class1_n/total
    class2_proportion = class2_n/total

    # implement entropy function
    return  sum([-1 * prop * np.log2(prop)
                 for prop in [class1_proprtion, class2_proportion] ])

print(entropy(3,1)) # --> 0.8112781244591328


### GRADED
### Code a function called `ent_from_split`
### ACCEPT three inputs:
### 1. A numpy array of values
### 2. A value on which to split the values in the first array, (into two groups; <= and >)
### 3. Labels for the observations corresponding to each value in the first array.
### ### Assume the labels are "0"s and "1"s

### RETURN the entropy resulting from that split: a float between 0 and 1.

### Feel free to use the `entropy()` function defined above
### YOUR ANSWER BELOW

def ent_from_split(col, split_value, labels):
    """
    Calculate the entropy of a split.
    
    Positional arguments:
        col -- a 1-dimensional numpy array, corresponding to a numeric
            predictor variable.
        split_value --  number, defining where the spliting should occur
        labels -- a 1-dimensional numpy array, corresponding to the class
            labels associated with the observations in `col`.
            assume they will be "0"s and "1"s
    Example:
        col = np.array([1,1,2,2,3,3,4])
        split = 2.5
        labels = np.array([0,1,0,0,1,0,1])        
        ent = ent_from_split(col, split, labels)
        print(ent) # --> 0.8571428571428571
    """
    idx_n1, idx_n2 = col <= split_value, col > split_value
    node1_x, node2_x = col[idx_n1], col[idx_n2]
    node1_y, node2_y = labels[idx_n1], labels[idx_n2]
    
    p_node1 = len(node1_x) / len(col)
    p_node2 = len(node2_x) / len(col)
    assert p_node1 + p_node2 == 1
    
    ent_node1 = entropy(sum(node1_y == 0), sum(node1_y == 1))
    ent_node2 = entropy(sum(node2_y == 0), sum(node2_y == 1))
    return float(p_node1*ent_node1 + p_node2*ent_node2)

col = np.array([1,1,2,2,3,3,4])
split = 2.5
labels = np.array([0,1,0,0,1,0,1])
ent = ent_from_split(col, split, labels)
print(ent) # --> 0.8571428571428571


# PART III - PREDICT FROM THE OBSERVED MAJORITY CLASS AT EACH NODE
# ----------------------------------------------------------------


### GRADED
### Code a function called `pred_from_split`
### ACCEPT four inputs:
### 1. a numpy array of observations
### 2. a numpy array of labels: 0's and 1's
### 3. a column index
### 4. a value to split that column specified by the index

### RETURN a tuple of (left_pred, right_pred) where:
### left_pred is the majority class of labels where observations are <= split_value
### right_pred is the majority class of labels where observations are > split_value

### If the split yeilds equal number of observations of each class in BOTH nodes,
### ### let both `left_pred` and `right_pred` be 1.
### If the split yeilds equal number of observations of each class in ONLY ONE node,
### ### predict the opposite of the other node. e.g.

### ### node 1    |   node 2
### ###  c1  | c2 |  c1 | c2
### ###  5  | 4   |  3  |  3

### The prediction for node 1 would be "class 1".
### Because of the equal numbers of each class in node 2,
### the prediction for node 2 would be the opposite of the node 1 prediction.
### e.g. the prediction for node 2 would be "class 2"

### YOUR ANSWER BELOW

def pred_from_split(X, y, col_idx, split_value):
    """
    Return predictions for the nodes defined by the given split.
    
    Positional argument:
        X -- a 2-dimensional numpy array of predictor variable observations.
            rows are observations, columns are features.
        y -- a 1-dimensional numpy array of labels, associated with observations
             in X.
        col_idx -- an integer index, such that X[:,col_idx] yeilds all the observations
            of a single feature.
        split_value -- a numeric split, such that the values of X[:,col_idx] that are
            <= split_value are in the left node. Those > split_value are in the right node.
    """    
    col = X[:,col_idx]
    idx_n1, idx_n2 = col <= split_value, col > split_value
    node1_y, node2_y = y[idx_n1], y[idx_n2]
    
    node1_c1, node1_c2 = sum(node1_y == 0), sum(node1_y == 1)
    node2_c1, node2_c2 = sum(node2_y == 0), sum(node2_y == 1)
    
    if node1_c1 == node1_c2:
        right_pred = 0 if node2_c1 > node2_c2 else 1
        left_pred = 1 - right_pred
    
    elif node2_c1 == node2_c2:
        left_pred = 0 if node1_c1 > node2_c2 else 1
        right_pred = 1 - left_pred
        
    else:
        left_pred = 0 if node1_c1 > node2_c2 else 1
        right_pred = 0 if node2_c1 > node2_c2 else 1
    
    return (left_pred, right_pred)


X = np.array([[0.5, 3. ], [1.,  2. ], [3.,  0.5], [2.,  3. ], [3.,  4. ]])    
y = np.array([ 1, 1, 0, 0, 1])
col_idx = 0
split_value = 1.5
pred_at_nodes = pred_from_split(X, y, col_idx, split_value)
print(pred_at_nodes) # --> (1, 0)


### GRADED
### Code a function called "simple_binary_tree_predict"
### ACCEPT five inputs:
### 1. A numpy array of observations
### 2. A column index
### 3. A value to split the column specified by the index
### 4/5. Two values, 1 or 0, denoting the predictions at left and right nodes

### RETURN a numpy array of predictions for each observation

### Predictions are created for each row in x:
### 1. For a row in X, find the value in the "col_idx" column
### 2. Compare to "split_value"
### 3. If <= "split_value", predict "left_pred"
### 4. Else predict "right_pred"x

### YOUR ANSWER BELOW

def simple_binary_tree_predict(X, col_idx, split_value, left_pred, right_pred):
    """
    Create an array of predictions built from: observations in one column of X,
        a given split value, and given predictions for when observations
        are less-than-or-equal-to that split or greater-than that split value
        
    Positional arguments:
        X -- a 2-dimensional numpy array of predictor variable observations.
            rows are observations, columns are different features
        col_idx -- an integer index, such that X[:,col_idx] yeilds all the observations
            in a single feature.
        split_value -- a numeric split, such that the values of X[:,col_idx] that are
            <= split_value are in the left node, and those > are in the right node.   
        left_pred -- class (0 or 1), that is predicted when observations
            are less-than-or-equal-to the split value
        right_pred -- class (0 or 1), that is predicted when observations
            are greater-than the split value
            
    """
    col = X[:,col_idx]
    idx_n1, idx_n2 = col <= split_value, col > split_value
    col[idx_n1], col[idx_n2] = left_pred, right_pred
    return col


X = np.array([[0.5, 3. ], [1.,  2. ], [3.,  0.5], [2.,  3. ], [3.,  4. ]])
col_idx = 0
split_value = 1.5
left_pred = 1
right_pred = 0
preds = simple_binary_tree_predict(X, col_idx, split_value, left_pred, right_pred)

print(preds) #--> np.array([1,1,0,0,0])


### ADABOOST
# ----------

### This helper function will return an instance of a `DecisionTreeClassifier` with
### our specifications - split on entropy, and grown to depth of 1.
from sklearn.tree import DecisionTreeClassifier

def simple_tree():
    return DecisionTreeClassifier(criterion = 'entropy', max_depth= 1)


### Our example dataset, inspired from lecture
pts = [[.5, 3,1],[1,2,1],[3,.5,-1],[2,3,-1],[3,4,1],
 [3.5,2.5,-1],[3.6,4.7,1],[4,4.2,1],[4.5,2,-1],[4.7,4.5,-1]]

df = pd.DataFrame(pts, columns = ['x','y','classification'])

# Plotting by category

b = df[df.classification ==1]
r = df[df.classification ==-1]
plt.figure(figsize = (4,4))
plt.scatter(b.x, b.y, color = 'b', marker="+", s = 400)
plt.scatter(r.x, r.y, color = 'r', marker = "o", s = 400)
plt.title("Categories Denoted by Color/Shape")
plt.show()


print("df:\n",df, "\n")

### split out X and y
X = df[['x','y']]

# Change from -1 and 1 to 0 and 1
y = np.array([1 if x == 1 else 0 for x in df['classification']])

### Split data in half
X1 = X.iloc[:len(X.index)//2, :]
X2 = X.iloc[len(X.index)//2:, :]

y1 = y[:len(y)//2]
y2 = y[len(X)//2:]


### Fit classifier to both sets of data, save to dictionary:

tree_dict = {}

tree1 = simple_tree()
tree1.fit(X1,y1)
print("threshold:", tree1.tree_.threshold[0], "feature:", tree1.tree_.feature[0])

### made up alpha, for example
alpha1 = .6
tree_dict[1] = (tree1, alpha1)

tree2 = simple_tree()
tree2.fit(X2,y2)
print("threshold:", tree2.tree_.threshold[0], "feature:" ,tree2.tree_.feature[0])

### made up alpha, again.
alpha2 = .35

tree_dict[2] = (tree2, alpha2)

### Create predictions using trees stored in dictionary
print("\ntree1 predictions on all elements:", tree_dict[1][0].predict(X))
print("tree2 predictions on all elements:", tree_dict[2][0].predict(X))

### Showing Ent
print("\nEntropy of different splits for observations 5-9")
print("Col 1, @ 3.35:", ent_from_split(X2.iloc[:,1].values,3.35, y2))
print("Col 0, # 4.25:", ent_from_split(X2.iloc[:,0].values, 4.25, y2))


      
### ADABOOST
### --------

'''
Building Adaptive Boosting
Below Gives the outline of the fitting process for the adaptive boosting algorithm.

Again, the functions next to "###<------" will be created in the exercises below. They include:

default_weights()
calc_epsilon()
calc_alpha()
update_weights()
'''

def simple_adaboost_fit(X,y, n_estimators):
    """
    Positional arguments :
        X -- a numpy array of numeric observations:
            rows are observations, columns are features
        y -- a numpy array of binary labels:
            *Assume labels are 1 for "True" and 0 for "False"*
        estimator -- a model capable of binary classification, implementing
            the `.fit()` and `.predict()` methods.
        n_estimators -- The number of estimators to fit.

    Steps:
        1. Create probability weights for selection during boot-straping.
        2. Create boot-strap sample of observations according to weights
        3. Fit estimator model with boot-strap sample.
        4. Calculate model error: epsilon
        5. Calculate alpha to associate with model
        6. Re-calculate probability weights
        7. Repeat 2-6 unil creation of n_estimators models. 

    """
    
    def simple_tree():
        return DecisionTreeClassifier(criterion = 'entropy', max_depth= 1)
    
    # Create default weights array where all are equal to 1/n
    weights = default_weights(len(y)) ### <------
    
    est_dict = {}
    for i in range(n_estimators):
        # Create bootstrap sample
        bs_X, bs_y = boot_strap_selection(X, y, weights)
        
        mod = simple_tree()
        mod.fit(bs_X, bs_y)
        
        # Note: Predicting on all values of X, NOT boot-strap
        preds = mod.predict(X)
        
        epsilon = calc_epsilon(y, preds, weights) ### <------
        alpha = calc_alpha(epsilon) ### <------
        
        # Note that the i+1-th model will be keyed to the int i,
        # and will store a tuple of the fit model and the alpha value
        est_dict[i] = (mod, alpha)
        
        weights = update_weights(weights, alpha, y, preds) ### <------
    
    return est_dict       
      
      

def default_weights(n):
    """
    Create the default list of weights, a numpy array of length n
    with each value equal to 1/n        
    """
    return np.array([1/n for _ in range(n)])

n = 10
dw = default_weights(n)
print(dw) #--> np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])


def boot_strap_selection(X, y, weights):
    """
    Create and return a boot-strapped sample of the given data,
    According to the provided weights.
    
    Positional Arguments:
        X -- a numpy array, corresponding to the matrix of x-observations
        y -- a numpy array, corresponding to a vector of y-labels
            All either 0 or 1
        weights -- a numpy array, corresponding to the rate at which the observations
            should be sampled for the boot-strap. 
    """
    # Take random sample of indicies, with replacement
    bss_indicies = np.random.choice(range(len(y)), size = len(y), p = weights)
    
    # Subset arrays with indicies
    return X[bss_indicies,:], y[bss_indicies]

### Example of use
X = np.array([[1,1],[2,2],[3,3],[4,4],[5,5]])
y = np.array([1,0,1,0,1])
weights = np.array([.35,.1,.1,.35,.1])

print(boot_strap_selection(X,y, weights))    


def calc_epsilon(y_true, y_pred, weights):
    
    """
    Calculate the value of epsilon, given the above equation 
    
    Positional Arguments:
        y_true -- An np.array of 1's and 0's corresponding to whether each observation is
            a member of class 1 or class 2
        y_pred -- An np.array of 1's and 0's corresponding to whether each observation was
            predicted to be a member of class 1 or class 2
        weights -- An np.array of floats corresponding to each observation's weight. 
            All the weights will sum up to 1.
        
    Assumptions:
        Assume both the true labels and the predictions are both all 0's and 1's.
    """
    return float(np.dot(weights, y_pred == y_true))
    
y_true = np.array([1,0,1,1,0])
y_pred = np.array([0,0,0,1,0])
weights = np.array([.4,.4,.1,.05,.05])

ep = calc_epsilon(y_true, y_pred, weights)

print(ep) # --> .5



def calc_alpha(epsilon):
    """
    Calculate the alpha value given the epsilon observed from a model
    
    Positional Argument:
        epsilon -- The epsilon value calculated from a particular model
    """    
    return float(0.5 * np.log((1-epsilon)/epsilon))

ep = .4
alpha = calc_alpha(ep)
print(alpha) # --> 0.2027325540540821


def update_weights(weights, alpha, y_true, y_pred):
    """
    Create an updated vector of weights according to the above equations    
    Positional Arguments:
        weights -- a 1-d numpy array of positive floats, corresponding to 
            observation weights
        alpha -- a positive float
        y_true -- a 1-d numpy array of true labels, all 0s and 1s
        y_pred -- a 1-d numpy array of labels predicted by the last model;
    """
    def change_labels(arr):
        for i,a in enumerate(arr):
            if a == 0:
                arr[i] = -1
        return arr        
            
    y_true, y_pred = change_labels(y_true), change_labels(y_pred)
    w_hat = weights * np.exp(-alpha * y_true * y_pred)
    return w_hat / sum(w_hat)

y_true = np.array([1,0,1,1,0])
y_pred = np.array([0,0,1,1,1])
weights = np.array([.4,.4,.1,.05,.05])
alpha = 0.10033534773107562

print(update_weights(weights, alpha, y_true, y_pred))
#-->np.array([0.44444444 0.36363636 0.09090909 0.04545455 0.05555556])


def predict(X, est_dict):
    """
    Create a np.array list of predictions for all of the observations in x,
    according to the above equation.
    
    Positional Arguments:
        X -- a 2-d numpy array of X observations. Features in columns, 
            observations in rows.
        est_dict -- a dictionary consists of keys 0 through n with tuples as values
            The tuples will be (<mod>, alpha), where alpha is a float, and 
            <mod> is a sklearn DecisionTreeClassifier

    Assumptions:
        The models in the `est-dict` tuple will return 0s and 1s.
            HOWEVER, the prediction equation depends upon predictions
            of -1s and 1s.
            FINALLY, the returned predictions should be 0s and 1s.            
    """
    
    def sign(arr):
        for i,a in enumerate(arr):
            if a >= 0:
                arr[i] = 1
            else:
                arr[i] = 0
        return arr
        
    def change_labels(arr, l1, l2):
        for i,a in enumerate(arr):
            if a == l1:
                arr[i] = l2
        return arr        
    
    total = 0    
    for (tree, alpha) in est_dict.values():
        total += alpha * change_labels(tree.predict(X), 0, -1)
        
    return sign(total)

### Our example dataset, inspired from lecture
pts = [[.5, 3,1],[1,2,1],[3,.5,0],[2,3,0],[3,4,1],
 [3.5,2.5,0],[3.6,4.7,1],[4,4.2,1],[4.5,2,0],[4.7,4.5,0]]

df = pd.DataFrame(pts, columns = ['x','y','classification'])

### split out X and labels
X = df[['x','y']]
y = df['classification']
### Split data in half
X1 = X.iloc[:len(X.index)//2, :]
X2 = X.iloc[len(X.index)//2:, :]

y1 = y[:len(y)//2]
y2 = y[len(X)//2:]


### Fit classifiers to both sets of data, save to dictionary:

### Tree-creator helper function
# def simple_tree():
#     return DecisionTreeClassifier(criterion = 'entropy', max_depth= 1)
    
tree_dict = {}

tree1 = simple_tree()
tree1.fit(X1,y1)
print("threshold:", tree1.tree_.threshold[0], "feature:", tree1.tree_.feature[0])

### made up alpha, for example
alpha1 = .6
tree_dict[1] = (tree1, alpha1)

tree2 = simple_tree()
tree2.fit(X2,y2)
print("threshold:", tree2.tree_.threshold[0], "feature:" ,tree2.tree_.feature[0])

### made up alpha, again.
alpha2 = .35
tree_dict[2] = (tree2, alpha2)

print(predict(X, tree_dict))
#--> np.array([1., 1., 0., 0., 0., 0., 0., 0., 0., 0.])

###############################
### For Further Checking of your function:
### The sum of predictions from the two models should be:

# If tree2 splits on feature 1:
# np.array([ 0.25  0.25 -0.95 -0.95 -0.25 -0.95 -0.25 -0.25 -0.95 -0.25])

# If tree2 splits on feature 0:
# np.array([ 0.95  0.95 -0.25 -0.25 -0.25 -0.25 -0.25 -0.25 -0.95 -0.95])
###############################



col_names = [
"age", "workclass", "fnlwgt", "education",
"education-num", "marital-status", "occupation", "relationship",
"race", "sex", "capital-gain", "capital-loss", "hours-per-week",
"native-country", "income"
]

data_path = "adult.data.txt"

data = pd.read_csv(data_path, header = None, names = col_names)
data.head()

# Take a subset
cols = ["age", "workclass", "education-num", "occupation", "sex", "hours-per-week", "income"]
data = data[cols]
data.head()

from sklearn.preprocessing import OneHotEncoder

ex_df = pd.DataFrame(
    np.array([['b', 'a', 'c', 'a', 'c',], ["z","y","y","y","z"]]).T,
    columns = ["beg","end"],
    index = range(500,505))

test_df = pd.DataFrame(
    np.array([["c","b"],["y","y"]]).T,
    columns = ["beg","end"],
    index = [56,72])

print("Initial DataFrame:")
print(ex_df)

print("\nTest df")
print(test_df)

# Instantiate OneHotEncoder
# sparse = False means data will not be stored in sparse matrix
ohe = OneHotEncoder(sparse = False)

# Fitting OHE with the "training" data
ohe.fit(ex_df)

# Transforming the "training" dat
tr_vals = ohe.transform(ex_df)

print("\nTransformed values")
print(tr_vals)

print("\nCategories")
print(ohe.categories_)

# Creating column names from `.categories_`
ohe_cats = np.concatenate(ohe.categories_)

# In creation of new df. Note the use of np.concatenate
final_df = pd.DataFrame(tr_vals, columns = ohe_cats)

print("\nFinal DataFrame")
print(final_df)

# Putting everything together to transform test data
print("\nTransformed test df")
print(pd.DataFrame(ohe.transform(test_df), columns= ohe_cats))



from sklearn.preprocessing import LabelEncoder

# Create target Series
target_train = pd.Series(np.random.choice(data['income'].unique(), size = 10))
target_test = pd.Series(np.random.choice(data['income'].unique(), size = 5))
print("Target Train Series")
print(target_train)

print("\nTarget Test Series")
print(target_test)

# Instantiate encoder
le = LabelEncoder()

# Fit with training data
le.fit(target_train)

# Transform training and test data
trans_train = le.transform(target_train)
trans_test = le.transform(target_test)

print("Transformed training values")
print(trans_train)

print("\nTransformed test values")
print(trans_test)

print("\nLabelEncoder `.classes_`")
print(le.classes_)



def preprocess_census(X_train, X_test, y_train, y_test):

    ### Hardcode variables which need categorical encoding
    to_encode = ["workclass", "occupation", "sex"]

    ### Find top categories in categorical columns
    ### Used for dropping majority class to prevent multi-colinearity
    top_categories = []

    for col in to_encode:
        top_categories.append(X_train[col].value_counts().index[0])

    ### Create and fit one-hot encoder for categoricals  
    OHE = OneHotEncoder(sparse = False)
    OHE.fit(X_train[to_encode])

    ## Create and fit Label encoder for target
    LabEnc = LabelEncoder()
    LabEnc.fit(y_train)

    def create_encoded_df(X, to_encode = to_encode, OHE = OHE, top_categories = top_categories):
        # Return columns which need encoding.
        def return_encoded_cols(X, to_encode = to_encode, OHE = OHE, top_categories = top_categories):
            # Use onehotencoder to transform.
            # Use "categories" to name
            toRet = pd.DataFrame(OHE.transform(X[to_encode]), columns = np.concatenate(OHE.categories_))

            # Drop top_categories and return
            return toRet.drop(top_categories, axis = 1)

        # create encoded columns
        ret_cols = return_encoded_cols(X)

        # Drop columns that were encoded
        dr_enc = X.drop(to_encode, axis = 1)

        # Concatenate values
        # use index from original data
        # use combined column names
        return pd.DataFrame(np.concatenate([ret_cols.values, dr_enc.values],axis = 1),
                            index = dr_enc.index,
                            columns = list(ret_cols.columns) + list(dr_enc.columns))


    def encode_target(y, LabEnc = LabEnc):
        # Use label encoder, and supply with original index
        return pd.Series(LabEnc.transform(y), index= y.index)

    return create_encoded_df(X_train), create_encoded_df(X_test), encode_target(y_train), encode_target(y_test)


from sklearn.model_selection import train_test_split

# Create training and testing sets; preprocess them.
target = data['income']
predictors = data.drop("income", axis = 'columns')

X_train, X_test, y_train, y_test = preprocess_census(*train_test_split(predictors, target, test_size = .2))



from sklearn.metrics import classification_report
d = simple_adaboost_fit(X_train.values.copy(), y_train.values.copy(), 5)
preds = predict(X_test, d)
print(classification_report(y_test,preds))