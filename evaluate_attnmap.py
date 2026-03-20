# alternative to evaluate.py, when heatmaps are needed. 
import torch
from torch import nn, Tensor
import math
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from scipy.io import savemat
import multiprocessing as mp
import argparse

from batchproc import get_batch
from transformermodelmixed import CustomTransformerMixedModel
import ppmmodel
from ctw import ctw_model
import config

index_count = 0

def ppm_findp(vocab_size, model, history, symbol):
	# Try to use highest order context that exists based on the history suffix, such
	# that the next symbol has non-zero frequency. 
    # note: slow implementation but only run in evaluation
    p_symbol = 1    
  
    for order in reversed(range(len(history) + 1)):
        ctx = model.root_context
        for sym in history[ : order]:
            assert ctx.subcontexts is not None
            ctx = ctx.subcontexts[sym]
            if ctx is None:
                break 
        else:  # ctx is not None
            if symbol != vocab_size and ctx.frequencies.get(symbol) > 0:
                p_sym = ctx.frequencies.get(symbol)/ctx.frequencies.get_total()
                p_symbol = p_symbol*p_sym
                break                        
            else: #Else write context escape symbol and continue decrementing the order
                p_esc = ctx.frequencies.get(vocab_size)/ctx.frequencies.get_total()                    
                p_symbol = p_symbol*p_esc                
    # Logic for order = -1 # 
    else:
        p_esc = 1/vocab_size
        p_symbol = p_symbol*p_esc
    
    return p_symbol 

def ppm_findp_mix(vocab_size, model, history, symbol):
    # mixing PPM models based on the error vector
    
    inf_val = -10e10
    almost_inf = -10e6
    depth = model.model_order + 1
    p_symbol = 1
    err = torch.ones(depth)*(inf_val)
    p_vec = torch.zeros(depth)
    increment_vec = torch.zeros(depth)
    p_symbol = 1      
    full_path = False

    if len(history)==model.model_order: 
        ctx = model.root_context
        for sym in history[ : model.model_order]:
            assert ctx.subcontexts is not None
            ctx = ctx.subcontexts[sym]
            if ctx is None:
                break 
        else:
            if ctx.frequencies.get(symbol) > 0: # run to full complete, and it has this symbol, take the mixing approach
                full_path = True
    
    if not full_path: # go with the classic PPM
        for order in reversed(range(len(history) + 1)):
            ctx = model.root_context
            for sym in history[ : order]:
                assert ctx.subcontexts is not None
                ctx = ctx.subcontexts[sym]
                if ctx is None:
                    break 
            else:  # ctx is not None
                if symbol != vocab_size and ctx.frequencies.get(symbol) > 0:
                    p_sym = ctx.frequencies.get(symbol)/ctx.frequencies.get_total()
                    p_symbol = p_symbol*p_sym                                           
                    break                        
                else: #Else write context escape symbol and continue decrementing the order
                    p_esc = ctx.frequencies.get(vocab_size)/ctx.frequencies.get_total()                    
                    p_symbol = p_symbol*p_esc
                    #comp_len[symbol] = comp_len[symbol] - np.log(p_esc)                                            

        # Logic for order = -1 # 
        else:
            p_esc = 1/vocab_size
            p_symbol = p_symbol*p_esc
    else: # go with the mix PPM        
        for order in reversed(range(0,len(history) + 1)):
            ctx = model.root_context
            for sym in history[ : order]:
                assert ctx.subcontexts is not None
                ctx = ctx.subcontexts[sym]
                if ctx is None:
                    break 
            else: # ctx is not None
                if order == model.model_order:
                    # read out the error vector
                    key_err = ctx.err # this copy we need to revise
                    err = ctx.err.clone() # this copy we'll need to use
                if symbol != vocab_size and ctx.frequencies.get(symbol) > 0:
                    total_freq_cl = ctx.frequencies.get_total() #-ctx.frequencies.get(model.escape_symbol)
                    p_vec[order] = ctx.frequencies.get(symbol)/total_freq_cl
                    #key_err[order] = err[order]
                    key_err[order] += torch.log(p_vec[order])
                #else: #Else won't use this context in producing p_symbol
                    #p_vec[order] = 0
                    # won't update err[order] in this case since we won't be using it

        # this is order -1 but no needed since this mode only invoked in full-path mode
        #p_vec[depth-1] = 1/vocab_size    
        #err[depth-1] = almost_inf        

        # now combine everything together
        p_symbol = torch.sum(torch.softmax(err*100,dim=0)*p_vec)

    return p_symbol 

def compute_singleCT_ppm_loss(args):
    sequence_index, eval_sequence, ground_truth, vocab_size, eval_bptt, sequence_len, num_segments, ppm_order, skip = args
  
    assert eval_bptt % num_segments == 0, "eval_seq_len must be divisible by num_segments"
    segment_length = int(eval_bptt/num_segments)
    groundtruth_rates = torch.zeros(num_segments)
    # prepare for the loss vectors
    PPM_loss = []
    for k in range(ppm_order):
        PPM_loss.append(torch.zeros(num_segments))
    mixPPM_loss = torch.zeros(num_segments)

    assert(math.floor(sequence_len/eval_bptt)*eval_bptt < sequence_len), "sequence_len must has at least one more symbol"

    for c in range(0,math.floor((sequence_len-1)/eval_bptt)): 
        pos = c*eval_bptt
        # reset the PPM models for each eval_bptt sequence
        # we have ppm_order+1 models: one for each order, then a mixPPM    
        PPM_histories = [] # mixPPM reused PPM3 history
        PPM_models = [] 
        PPM_tks = []        
        for k in range(ppm_order):
            PPM_histories.append([])
            PPM_models.append(ppmmodel.PpmModel(k+1, vocab_size+1, vocab_size))
            PPM_tks.append(torch.ones((eval_bptt+1))/vocab_size)
        PPM_modelmix = ppmmodel.PpmModel(ppm_order, vocab_size+1, vocab_size)        
        mixPPM_tk = torch.ones((eval_bptt+1))/vocab_size         
        
        # here is a single data sequence of bptt length
        subseq = eval_sequence[pos:pos+eval_bptt+1]
        subseq_ground_truth = ground_truth[pos:pos+eval_bptt+1]
        for j in range(eval_bptt+1):            
            next_symbol = subseq[j]
            mixPPM_tk[j] = ppm_findp_mix(vocab_size, PPM_modelmix, PPM_histories[ppm_order-1], next_symbol)            
            PPM_modelmix.increment_contexts(PPM_histories[ppm_order-1], next_symbol)    
            # maintain the PPM models, and obtain the estimated probilities
            p = torch.zeros(ppm_order)
            for k in range(ppm_order):
                p[k] = ppm_findp(vocab_size, PPM_models[k], PPM_histories[k], next_symbol)
                PPM_models[k].increment_contexts(PPM_histories[k], next_symbol)
                PPM_tks[k][j] = p[k]
                # also need to update the PPM history i.e., the state.
                # note that we need to do this after the p's have been computed, including the mixPPM
                if PPM_models[k].model_order >= 1: 
                    if len(PPM_histories[k]) == PPM_models[k].model_order:
                        PPM_histories[k].pop()
                    PPM_histories[k].insert(0, next_symbol)
                
        # now for this context window, we have the PPM probability estimate sequences
        # next we need to   the loss for each segment        
        # iterate over the segments        
        for s in range(num_segments):  
            # ground_truth already shifted by 1
            groundtruth_rates[s] += -torch.mean(torch.log(torch.clamp(subseq_ground_truth[s*segment_length:(s+1)*segment_length],min=1e-10)))
            # the prediction needs to be further shifted by 1
            for k in range(ppm_order):            
                PPM_loss[k][s] += -torch.mean(torch.log(torch.clamp(PPM_tks[k][s*segment_length+1:(s+1)*segment_length+1],min=1e-10)))
            mixPPM_loss[s] += -torch.mean(torch.log(torch.clamp(mixPPM_tk[s*segment_length+1:(s+1)*segment_length+1],min=1e-10)))
    
    #for k in range(ppm_order):
    #    PPM_loss[k] -= groundtruth_rates

    result_tensor = torch.stack(PPM_loss)
    result_tensor = torch.cat((result_tensor, mixPPM_loss.unsqueeze(0)),dim=0)    
    #result_tensor -= groundtruth_rates
    norm_factor = math.floor(sequence_len/eval_bptt)
    #result_tensor = result_tensor/math.floor(sequence_len/eval_bptt) # dim [batch_size, ppm_order+1, num_segments]    
    print(f"Complete context tree {sequence_index:3d}.",flush=True)
    return (result_tensor-groundtruth_rates)/norm_factor, result_tensor/norm_factor    # regrets and loss

# Function to generate data batch using multiprocessing
def ppm_evaluate_mp(eval_sequences: Tensor, ground_truth: Tensor, num_workers, vocab_size: int, eval_bptt:int, sequence_len: int, number_segments: int, ppm_order: int, skips):

    num_CTs = eval_sequences.size(1)
   
    with mp.Pool(processes=num_workers) as pool:
        # each worker is given one CT to process
        args = [(i, eval_sequences[:,i], ground_truth[:,i], vocab_size, eval_bptt, sequence_len, number_segments, ppm_order, skips[i]) for i in range(num_CTs)]
        results = pool.map(compute_singleCT_ppm_loss, args)
    
    # # # Separate tensors     
    # regrets, relative_regrets  = zip(*results)        
    # regrets_list = list(regrets)
    # relative_regrets_list = list(relative_regrets)
    # regrets_tensor = torch.stack(regrets_list)
    # relative_regrets_tensor = torch.stack(relative_regrets_list)

    # # PPM_loss_list = list(PPM_loss_list)
    # # data = torch.stack(PPM_loss_list)
    # # data = data.contiguous()    
    # # return data   
    # #data = torch.stack(results)    
    # return regrets_tensor, relative_regrets_tensor

    regrets, total_loss  = zip(*results)        
    regrets_list = list(regrets)
    loss_list = list(total_loss)
    regret_tensor = torch.stack(regrets_list)
    loss_tensor = torch.stack(loss_list)

    return regret_tensor, loss_tensor

def ctw_cal_rates(args, verbose=False):
    sequence_index, eval_sequence, ground_truth, M, D, beta, alpha, eval_bptt, sequence_len, num_segments, skip = args    
    CTW_loss = torch.zeros(num_segments)
    groundtruth_rates = torch.zeros(num_segments)

    # convert eval_sequence which is a vector of integers into a string  
    # unique_values = set(eval_sequence.tolist())
    # mapping = {val: str(val) for val in range(M)}      
    # Convert tensor to string tensor
    #eval_sequence  = ''.join([mapping[x.item()] for x in eval_sequence])
    eval_sequence = eval_sequence.tolist()

    assert(math.floor(sequence_len/eval_bptt)*eval_bptt < sequence_len), "sequence_len must has at least one more symbol"
    segment_length = math.floor(eval_bptt/num_segments)
    for c in range(0,math.floor((sequence_len-1)/eval_bptt)): 

        pos = c*eval_bptt
        # here is a single data sequence of bptt length
        historystate = eval_sequence[pos-D:pos]
        subseq = eval_sequence[pos:pos+eval_bptt+1]
        subseq_ground_truth = ground_truth[pos:pos+eval_bptt]
        log_p = torch.zeros(eval_bptt+1)
        log_p[:D] = -np.log(1/M) # initial prediction is uniform

        # note the prior value. For CTW to be Bayesian optimal, this should match the generating model
        CTW = ctw_model(M, D, beta, prior = alpha, seq = historystate) # create a CTW class, reset everything in CTW
        for j in range(0,eval_bptt+1):
            predict_prob, _ = CTW.predict()
            log_p[j] = -np.log(max(predict_prob[int(subseq[j])],1e-10))            
            CTW.update_seq(subseq[j])        
        # next we need to the loss for each segment        
        # iterate over the segments        
        for s in range(num_segments):    
            CTW_loss[s] += torch.mean(log_p[s*segment_length+1:(s+1)*segment_length+1])        
            # ground_truth already shifted by 1
            groundtruth_rates[s] += -torch.mean(torch.log(torch.clamp(subseq_ground_truth[s*segment_length:(s+1)*segment_length],min=1e-10)))

    #CTW_loss -= groundtruth_rates
    norm_factor = math.floor(sequence_len/eval_bptt)
    #result_tensor = CTW_loss/math.floor(sequence_len/eval_bptt) # dim [batch_size, ppm_order+1, num_segments]    
    print(f"Complete context tree {sequence_index:3d}.",flush=True)
    return (CTW_loss-groundtruth_rates)/norm_factor, CTW_loss/norm_factor    # regrets and relative regrets

# Function to generate data batch using multiprocessing
def ctw_evaluate_mp(eval_sequences: Tensor, ground_truth: Tensor, num_workers, vocab_size: int, eval_bptt:int, sequence_len: int, number_segments: int, ct_depth: int, 
                    beta, alpha, skips):

    num_CTs = eval_sequences.size(1)
    
    with mp.Pool(processes=num_workers) as pool:
        # each worker is given one CT to process
        args = [(i, eval_sequences[:,i], ground_truth[:,i], vocab_size, ct_depth, beta, alpha, eval_bptt, sequence_len, number_segments, skips[i]) for i in range(num_CTs)]
        results = pool.map(ctw_cal_rates, args)
    

    regrets, total_loss  = zip(*results)        
    regrets_list = list(regrets)
    loss_list = list(total_loss)
    regret_tensor = torch.stack(regrets_list)
    loss_tensor = torch.stack(loss_list)

    return regret_tensor, loss_tensor


def get_attn_weights(model: nn.Module, data, result_dir: str):
    
    model.eval()  # turn on eval mode    
    with torch.no_grad():        
        if model.n_layers==0:
            print('Zero-layer model does not have attention weights.')
            return
        output = model(data.unsqueeze(1))                
        # extracting attension weights        
        attention_weights = model.transformer_encoder.get_attention_weights()
        sizes = [tensor.size() for tensor in attention_weights]
        print(sizes)
        # Visualizing the attention weights        
        result_file = Path(os.path.join(result_dir, f"data{index_count}.mat"))
        #index_count = index_count + 1
        attention_weights_cpu = [tensor.cpu() for tensor in attention_weights]        
        data_cpu = data.cpu()
        savemat(result_file,{'attention_weights':attention_weights_cpu,'data':data_cpu})
        
        for i, layer in enumerate(model.transformer_encoder.layers):
            for j in range(layer.self_attn.num_heads):
                # Plot the attention map
                plt.figure(figsize=(10, 8))          
                attention_weights_single_layer = attention_weights[i][0,j,:,:].detach().cpu().numpy()
                plt.imshow(attention_weights_single_layer[:63,:63], cmap='Purples')
                if i==0 and j==5:
                    cbar = plt.colorbar()
                    cbar.ax.tick_params(labelsize=16)  # Adjust colorbar font size
                plt.xlabel('Key Position', fontsize=26)
                plt.ylabel('Query Position',fontsize=26)
                #plt.title(f"Attention Weights for Layer-{i+1}",fontsize=18)
                # Adjust x-axis values font size
                plt.xticks(fontsize=26)
                # Adjust y-axis values font size
                plt.yticks(fontsize=26)
                # Save the attention map to a PDF file          
                image_file = Path(os.path.join(result_dir, f"attn_layer{i+1}_head{j+1}.pdf"))    
                plt.savefig(image_file)
                plt.close()


def eval_batch(model: nn.Module, eval_data: Tensor, ground_truth: Tensor, bptt:int, num_segments: int):
   
    criterion = nn.CrossEntropyLoss()
    model.eval()  # turn on eval mode
    total_loss = torch.zeros(eval_data.size(1),num_segments)  # Initialize total_loss as a tensor of zeros [batch_size,num_segments]
    groundtruth_rates = torch.zeros(eval_data.size(1),num_segments)
    assert bptt % num_segments == 0, "eval_seq_len must be divisible by num_segments"
    segment_length = int(bptt/num_segments)
    eval_seq_len, _ = eval_data.size() # the sequence length, k is the number of contexts, l is the batch size
    
    with torch.no_grad():
        num_blocks = math.floor((eval_seq_len-1)/bptt)
        for i in range(num_blocks):      
            pos = i*bptt
            data, targets = get_batch(eval_data, pos, bptt)    # [bptt, batch_size], and bptt*batch_size, respectively                  
            output = model(data)
            [seq_len, batch_size] = data.size()            
            subseq_ground_truth = ground_truth[pos:pos+bptt+1,:]

            targets = targets.view(seq_len, batch_size)

            # iterate over the segments
            for k in range(num_segments):
                output_part = output[k*segment_length:(k+1)*segment_length]                
                targets_part = targets[k*segment_length:(k+1)*segment_length]

                for j in range(batch_size):
                    output_batch = output_part[:, j, :]  # Select the j-th batch element
                    targets_batch = targets_part[:, j]   # Select the j-th batch element
                    total_loss[j,k] += criterion(output_batch, targets_batch).item()
                    # ground_truth is already shifted by 1, so no need to shift again here
                    # ground_truth is a tensor of [eval_seq_len, batch_size]
                    groundtruth_rates[j,k] += -torch.mean(torch.log(torch.clamp(subseq_ground_truth[k*segment_length:(k+1)*segment_length,j],min=1e-10))) 


    norm_factor = math.floor(eval_seq_len/bptt)     
    regrets = total_loss - groundtruth_rates
    #relative_regrets = regrets/groundtruth_rates    
    return regrets/norm_factor, total_loss/norm_factor #relative_regrets/norm_factor

def evaluate_model(cfg):

    vocab_size = cfg.vocab_size    
    # evalation dataset parameters        
    eval_max_num_skip = cfg.eval_max_num_skip
    eval_max_tree_depth = cfg.eval_max_tree_depth
    eval_mix_skip = cfg.eval_mix_skip
    eval_mix_depth = cfg.eval_mix_depth
    eval_data_seq_len = cfg.eval_data_seq_len
    num_eval_CTs = cfg.num_eval_CTs    
    eval_bptt = cfg.eval_bptt
    eval_mode = cfg.eval_mode # 'TF', 'PPM', 'CTW'
    eval_data_seq_selection_len = cfg.eval_data_seq_selection_len
    number_segments = cfg.number_segments
    eval_p = cfg.eval_tree_nodesplit_p
    eval_alpha = cfg.eval_alpha
    
    # evaluation datasets
    data_dir = Path(cfg.datapath)            
    eval_file_name = f"eval_dataset_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.pt"    
    eval_file_path = data_dir/eval_file_name     
    # only test to see if the data files already exist.
    if eval_file_path.exists():
        print(f"Evaluation dataset exists at {eval_file_path}.")
        eval_data_loaded = torch.load(eval_file_path,weights_only=False)                
        print(f"Evaluation dataset loaded from {eval_file_path}.")
    else:
        print(f"Evaluation dataset not found at {eval_file_path}. Need to create new dataset. Exiting.")    
        return             
        
    # Extract tensors from eval_data
    eval_sequences = eval_data_loaded['data']    
    eval_p_tensor = eval_data_loaded['p_tensor']
    eval_rates = eval_data_loaded['rates']        
    eval_num_contexts = eval_data_loaded['num_contexts']
    eval_depths = eval_data_loaded['depths']
    eval_skips = eval_data_loaded['skips']

    n,_ = eval_sequences.size() # n is the sequence length, k is the number of contexts, l is the batch size
    ground_truth = eval_p_tensor[torch.arange(n)[:,None],torch.arange(num_eval_CTs), eval_sequences]
    #ground_truth = torch.ones(n,num_eval_CTs) # dummy ground truth for now

    
    # evaluations are different for transformer models and PPM algorithms
    if eval_mode == 'TF':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device: ', device)
        # if TF model is used, we need to evaluate on GPU
        eval_sequences = eval_sequences.contiguous().to(device)
        # parameters for the trained model to be evaluated
        trained_bptt = cfg.bptt    
             
        nheads = cfg.nheads  
        emsize = cfg.emsize    
        nlayers = len(nheads)    
        n_syn_layers = cfg.num_synthetic_layers
        dropout = cfg.dropout     
        max_num_skip = cfg.max_num_skip
        max_tree_depth = cfg.max_tree_depth  # the max context length is  (tree_depth - 1)            
        mix_skip = cfg.mix_skip
        mix_depth = cfg.mix_depth
        batch_size = cfg.eval_batch_size
        d_hid = cfg.d_hid
        p = cfg.tree_nodesplit_p
        alpha = cfg.eval_alpha


        model = CustomTransformerMixedModel(vocab_size, max_tree_depth,emsize, nheads, d_hid, nlayers, n_syn_layers ,dropout).to(device)          
        # load the trained model
        model_dir = Path(cfg.modelpath)
        nheads_string = "_".join([str(item) for item in nheads])  # This joins the items with underscores
        # standard model path
        best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt")            
        # nopos model path
        #best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos1.pt")            
        best_model_params_path = Path(best_model_params_path)    

        # load transformer model matching the specification in config.py 
        if best_model_params_path.exists():
            print(f"Model exists at {best_model_params_path}.")
            model.load_state_dict(torch.load(best_model_params_path,weights_only=False))
            model = model.to(device)
            print(f"Loaded model from {best_model_params_path}.")
        else:
            print(f"Cannot find model at {best_model_params_path}. Exiting")
            return
        
        result_dir = Path(cfg.resultpath)    
        # dir name shows the tf parameters
        # standard  path        
        result_dir = Path(os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}"))
        # nopos path
        #result_dir = Path(os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos1"))
        result_dir.mkdir(parents = True, exist_ok = True)   
        regret_store = torch.zeros(num_eval_CTs,number_segments)
        loss_store = torch.zeros(num_eval_CTs,number_segments)
        # file name shows the evaluation parameter        
    
        # get attention weights and plot        
        # let's find a position with representative data
        # this model will do the weights
        model = CustomTransformerMixedModel(vocab_size, max_tree_depth, emsize, nheads, d_hid, nlayers, n_syn_layers ,dropout, True).to(device) 
        model.load_state_dict(torch.load(best_model_params_path,weights_only=False))
        model = model.to(device)         

        CT_index =  5
        print(eval_num_contexts[CT_index])
        get_attn_weights(model, eval_sequences[:eval_bptt,CT_index], result_dir)

    return
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ICL of context trees")
    parser.add_argument('--config', type=str, required=True, help='Name of the configuration class in config.py')    
    args = parser.parse_args()

    config_class = getattr(config, args.config)
    cfg = config_class()
    
    evaluate_model(cfg)   
   
    

        
