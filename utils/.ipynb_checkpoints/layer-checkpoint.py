import numpy as np

def logit(array):
    return [1/(1+np.exp(-i)) for i in array]

def softmax(array):
    return [i/np.sum(logit(array)) for i in logit(array)]