import numpy as np
from utils.layer import softmax

def epsilon_greedy(state_action, epsilon=0.2):
    """
    state_action = current state and possible actions
    epsilon 
    """
    action_possible = [i for i in range(len(state_action))]
    action_probability = softmax(state_action)
    rnd_num = np.random.rand(1)
    
    if rnd_num < epsilon:
        action = np.random.choice(a=action_possible)
    else:
        action = np.random.choice(a=action_possible, p=state_action)
    return action