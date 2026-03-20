import torch
from pathlib import Path
import random
import numpy as np
import multiprocessing as mp
import time
import ctw

from random_tree import generate_tree

def generate_datapiece(args):
    MKmodel, context_dists, sequence_len, vocab_size, num_skip, skip_p = args
    # in this format so that it also be called by MP easily
    # here we generate a vector of length sequence_len
    # MKmodel: tree source model that generates the data sequence
    # note: we don't track the probability here in order to save on computation here
    
    state = []
    uniform_p = torch.tensor([1/vocab_size for _ in range(vocab_size)])    
    max_ctx_len = max(len(t) for t in MKmodel) + num_skip
    compress_length = 0 
    pos = 0
    buffer_size = 500
    sequence_len = buffer_size + sequence_len
    sequence = torch.zeros(sequence_len, dtype=torch.long)
    
    # to avoid if else in the loop, we put the skip_p condition outside the loop
    if skip_p:
        for _ in range(sequence_len):
            next_symbol = -1
            state_t = state[num_skip:] # temperary state to process

            while len(state_t)>0:          
                state_tuple = tuple(state_t)  
                if state_tuple in MKmodel:
                    index = MKmodel.index(state_tuple)
                    p = context_dists[index]                       
                    next_symbol = np.random.choice(vocab_size, p=p)   
                    compress_length += - np.log(p[next_symbol])                                 
                    # found the next_symbol, so we can break
                    break
                else:
                    # maybe the state_t state is too large, but only a shorter index exists
                    state_t = state_t[:-1]                
                
            if next_symbol == -1:
                # if not in the model (because the context is too short, i.e., at the begining), uniformly generate a symbol
                if len(MKmodel)==1:
                    p = context_dists[0]                       
                    next_symbol = np.random.choice(vocab_size, p=p)   
                    compress_length += - np.log(p[next_symbol])       
                    
                else:
                    next_symbol = random.choice(range(vocab_size))             
                    compress_length += np.log(vocab_size)                    

            sequence[pos] = next_symbol
            pos += 1
            state.insert(0, next_symbol)
            state = state[:max_ctx_len]
        rate = compress_length/sequence_len # this rate is not really used in the training, but useful in validation and evaluation    
        return sequence[buffer_size:], rate      
    else:
        p_sequence = torch.zeros(sequence_len, vocab_size)
        for _ in range(sequence_len):
            next_symbol = -1
            state_t = state[num_skip:] # temperary state to process

            while len(state_t)>0:          
                state_tuple = tuple(state_t)  
                if state_tuple in MKmodel:
                    index = MKmodel.index(state_tuple)
                    p = context_dists[index]                       
                    next_symbol = np.random.choice(vocab_size, p=p)   
                    compress_length += - np.log(p[next_symbol])             
                    p_sequence[pos,:] = p
                    # found the next_symbol, so we can break
                    break
                else:
                    # maybe the state_t state is too large, but only a shorter index exists
                    state_t = state_t[:-1]                
            
            # if not in the model (because the context is too short or the model is zero-th order, i.e., at the begining),
            if next_symbol == -1:
                # zero-th order model                
                if len(MKmodel)==1:
                    p = context_dists[0]                       
                    next_symbol = np.random.choice(vocab_size, p=p)   
                    compress_length += - np.log(p[next_symbol])       
                    p_sequence[pos,:]=p      
                #  uniformly generate a symbol
                else:
                    next_symbol = random.choice(range(vocab_size))             
                    compress_length += np.log(vocab_size)
                    p_sequence[pos,:]=uniform_p

            sequence[pos] = next_symbol
            pos += 1
            state.insert(0, next_symbol)
            state = state[:max_ctx_len]
        rate = compress_length/sequence_len # this rate is not really used in the training, but useful in validation and evaluation        
        return sequence[buffer_size:], rate, p_sequence[buffer_size:,:]

def generate_single_data(args):
    i, vocab_size, sequence_len, depth, skip, p, alpha, skip_p = args    
    
    MKmodel, context_dists = generate_tree(vocab_size, depth, p, alpha)   
    print(f'Generating context tree {i}: depth {max(len(model) for model in MKmodel)}, num_Cts {len(MKmodel)}')    

    if skip_p:
        new_tensor, rate = generate_datapiece([MKmodel, context_dists, sequence_len, vocab_size, skip, skip_p])                
        return new_tensor, rate, len(context_dists)            
    else:
        new_tensor, rate, p_sequence = generate_datapiece([MKmodel, context_dists, sequence_len, vocab_size, skip, skip_p])        
        return new_tensor, rate, p_sequence, len(context_dists)



# Function to generate data batch using multiprocessing
def generate_databatch_mp(num_workers, vocab_size, sequence_len: int, depths, skips, p, alpha, num_CTs, skip_p):
  
    with mp.Pool(processes=num_workers) as pool:
       args = [(i, vocab_size, sequence_len, depths[i], skips[i], p, alpha, skip_p) for i in range(num_CTs)]        
       results = pool.map(generate_single_data, args)
    
    # Separate tensors and rates
    if skip_p:
        tensor_list, rate_list, length_list = zip(*results)
    else:
        tensor_list, rate_list, p_list, length_list = zip(*results)
    tensor_list = list(tensor_list)
    rate_list = list(rate_list)
    length_list  = list(length_list)
    data = torch.stack(tensor_list)
    data = data.t().contiguous()
    rate_tensor = torch.tensor(rate_list)    
    length_tensor = torch.tensor(length_list) 

    if skip_p:
        return data, rate_tensor, length_tensor
    else:
        p_list = list(p_list)
        p_tensor = torch.stack(p_list)
        p_tensor = p_tensor.permute(1,0,2).contiguous() # [length, num_CTs, vocab_size]
        return data, rate_tensor, p_tensor, length_tensor
    
    # with mp.Pool(processes=num_workers) as pool:
    #     args = [(i, 0, vocab_size, depths[i], 1-p, alpha, sequence_len) for i in range(num_CTs)]        
    #     results = pool.map(ctw.generate_seq, args)
    
    # data = torch.stack(list(results))  
    # return data.transpose(0,1).contiguous() 


def data_gen(cfg):
    generate_data('train',cfg)
    generate_data('val',cfg)
    generate_data('eval',cfg)

def generate_data(mode,cfg):

    vocab_size = cfg.vocab_size            
    data_dir = Path(cfg.datapath)  
    num_workers = cfg.num_processes

    if(mode == 'train' or mode == 'val'):
        max_num_skip = cfg.max_num_skip                    # the max number of skips 
        max_tree_depth = cfg.max_tree_depth                # the max context length is equal to tree_depth     
        mix_depth = cfg.mix_depth                          # whether to mix depths
        mix_skip = cfg.mix_skip                            # whether to mix skips
        p = cfg.tree_nodesplit_p
        alpha = cfg.alpha
          
        if mode == 'train':
            num_CTs = cfg.num_train_CTs
            sequence_len = cfg.training_data_seq_len
            file_name = f"train_dataset_v{vocab_size}_d{max_tree_depth}_s{max_num_skip}_{sequence_len}_{num_CTs}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt"                            
        else:            
            num_CTs = cfg.num_val_CTs
            sequence_len = cfg.val_data_seq_len
            file_name = f"val_dataset_v{vocab_size}_d{max_tree_depth}_s{max_num_skip}_{sequence_len}_{num_CTs}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt"                    

    elif mode == 'eval':
        max_num_skip = cfg.eval_max_num_skip                # the max number of skips 
        max_tree_depth = cfg.eval_max_tree_depth            # the max context length is equal to tree_depth
        mix_depth = cfg.eval_mix_depth                      # whether to mix depths
        mix_skip = cfg.eval_mix_skip                        # whether to mix skips
        p = cfg.eval_tree_nodesplit_p
        alpha = cfg.eval_alpha

        num_CTs = cfg.num_eval_CTs
        sequence_len = cfg.eval_data_seq_len
        file_name = f"eval_dataset_v{vocab_size}_d{max_tree_depth}_s{max_num_skip}_{sequence_len}_{num_CTs}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt"
    file_path = data_dir/file_name

    start_time = time.time()  # Start timing
    # test to see if the data file already exists.
    if file_path.exists():
        print(f"Dataset already exists at {file_path}.")    
    else:
        print(f"Dataset not found at {file_path}. Creating new dataset.")    
        # Ensure the data directory exists
        data_dir.mkdir(parents=True, exist_ok=True)   
        skips = [random.randint(0, max_num_skip) for _ in range(num_CTs)] if mix_skip else  [max_num_skip for _ in range(num_CTs)]
        depths = [random.randint(0, max_tree_depth) for _ in range(num_CTs)] if mix_depth else [max_tree_depth for _ in range(num_CTs)]
        
        if mode == 'eval':
            data,rates, p_tensor, lengths = generate_databatch_mp(num_workers, vocab_size, sequence_len, depths, skips, p, alpha, num_CTs, False)
            #data = generate_databatch_mp(num_workers, vocab_size, sequence_len, depths, skips, p, alpha, num_CTs, False)
            torch.save({'data':data, 'rates':rates, 'p_tensor': p_tensor ,'num_contexts':lengths, 'depths':depths, 'skips':skips, 'p':p, 'alpha':alpha}, file_path)
            #torch.save({'data':data}, file_path)
        else:
            data,rates, lengths = generate_databatch_mp(num_workers, vocab_size, sequence_len, depths, skips, p, alpha, num_CTs, True)
            torch.save({'data':data, 'rates':rates, 'num_contexts':lengths, 'depths':depths, 'skips':skips, 'p': p, 'alpha':alpha}, file_path)
        # Save the dataset
        print(f"Dataset saved at {file_path}.")
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    print(f"Data generation took {elapsed_time:.2f} seconds.")