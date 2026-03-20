from random import random, uniform
from math import ceil,floor
import numpy as np
import torch

num_leaf = 0

def rand_tree(prefixes, p, depth, max_depth, vocab_size, prefix):
    # recursively construct a random tree that at each node, either branch into num_vocab children, or stay as it.
    # Note: This is only one way to generate the tree: starting from the root and see if we split or not to produce children

    # prefixes: the list of prefixes/suffixes 
    # p: the probability that a node is branching
    # depth: the current depth
    # max_depth: the maximum depth allowed for the tree source model
    # vocab_size: how large the vocabulary is
    
    # if at the bottom, add the prefix into the list, then return
    if depth == max_depth:
        prefixes.append(prefix[1:])
        return
    
    if p >= random():
        for i in range(vocab_size):            
            rand_tree(prefixes, p, depth+1, max_depth, vocab_size, prefix+(i,))            
    else:
        prefixes.append(prefix[1:])
        return
            
def generate_tree(vocab_size,max_depth, p, alpha): # here max_depth is the deepest the tree can go, but due to random selection, is might be shallower        

    # initialization        
    prefixes = []
    context_dists = []    

    # random tree generation       
    rand_tree(prefixes, p, 0, max_depth, vocab_size, (vocab_size,))
    prefixes.sort()     # this is a sorted list, therefore the search in the list is of log(n), which is the depth of the tree
    num_zeros = vocab_size - max(floor(vocab_size/2),2)
    if alpha<=0:        #  use sparse model
         for prefix in prefixes:
            probabilities = [uniform(0, 1) for _ in range(vocab_size)]
            probabilities = np.array(probabilities)
            probabilities = np.exp(probabilities*5)
            zero_indices = np.random.choice(vocab_size, num_zeros, replace=False)        
            probabilities[zero_indices] = 0 # we assign some zeros to make the in-context learning more explicit 
            probabilities = probabilities/sum(probabilities) 
            context_dists.append(torch.tensor(probabilities))            
    else:              # use Dirichlet distribution
        alpha = [alpha] * vocab_size  # This creates a list of 'n' elements, each equal to alpha_value
        for prefix in prefixes:                                
            context_dists.append(torch.tensor(np.random.dirichlet(alpha)))
    
    return [prefixes,context_dists]
            