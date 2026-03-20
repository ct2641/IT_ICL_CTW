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


from batchproc import get_batch
from transformermodelmixed import CustomTransformerMixedModel
import ppmmodel
from ctw import ctw_model

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

def kn_findp(vocab_size, model, history, symbol, delta):
    # Kneser-Ney smoothing
    p_symbol = 0
    weight = 1
    for order in reversed(range(len(history) + 1)):
        ctx = model.root_context
        for sym in history[ : order]:
            assert ctx.subcontexts is not None
            ctx = ctx.subcontexts[sym]
            if ctx is None:  # suffice has not appears previously,                                               
                break
        else: # suffix has appeard before, i.e., found it
            if symbol != vocab_size and ctx.frequencies.get(symbol) > delta:
                # only when have enough occurances, we use it in the weighted sum
                # note the -1 because of the escape symbol
                p_sym = (ctx.frequencies.get(symbol)-delta)/(ctx.frequencies.get_total()-1)
                p_symbol = p_symbol+p_sym*weight
            # Regardless whether the current estimate is used, we need to update the weight
            count = sum(ctx.frequencies.get(i) >0 for i in range(vocab_size))
            if count > 0 :
                # note the -1 because of the escape symbol
                weight = weight*delta/(ctx.frequencies.get_total()-1)*count                            
            
                            
    # finally, it always add the -1 order estimate
    p_symbol = p_symbol + weight/vocab_size
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
        subseq_ground_truth = ground_truth[pos+1:pos+eval_bptt+1]
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
        # next we need to  the loss for each segment        
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

def compute_singleCT_kn_loss(args):
    sequence_index, eval_sequence, ground_truth, vocab_size, eval_bptt, sequence_len, num_segments, kn_order, kn_delta, skip = args
    
    assert eval_bptt % num_segments == 0, "eval_seq_len must be divisible by num_segments"
    segment_length = int(eval_bptt/num_segments)
    groundtruth_rates = torch.zeros(num_segments)
    # prepare for the loss vectors
    kn_loss = []
    for k in range(kn_order):
        kn_loss.append(torch.zeros(num_segments))

    assert(math.floor(sequence_len/eval_bptt)*eval_bptt < sequence_len), "sequence_len must has at least one more symbol"

    for c in range(0,math.floor((sequence_len-1)/eval_bptt)): 
        pos = c*eval_bptt
        # reset the PPM models for each eval_bptt sequence
        # we have ppm_order+1 models: one for each order, then a mixPPM    
        kn_histories = [] 
        kn_models = [] 
        kn_tks = []        
        for k in range(kn_order):
            kn_histories.append([])
            kn_models.append(ppmmodel.PpmModel(k+1, vocab_size+1, vocab_size))
            kn_tks.append(torch.ones((eval_bptt+1))/vocab_size)
        
        # here is a single data sequence of bptt length
        subseq = eval_sequence[pos:pos+eval_bptt+1]
        subseq_ground_truth = ground_truth[pos:pos+eval_bptt]
        for j in range(eval_bptt+1):            
            next_symbol = subseq[j]
            p = torch.zeros(kn_order)
            for k in range(kn_order):
                p[k] = kn_findp(vocab_size, kn_models[k], kn_histories[k], next_symbol, kn_delta)
                kn_models[k].increment_contexts(kn_histories[k], next_symbol)
                kn_tks[k][j] = p[k]
                # also need to update the kn history i.e., the state.
                if kn_models[k].model_order >= 1: 
                    if len(kn_histories[k]) == kn_models[k].model_order:
                        kn_histories[k].pop()
                    kn_histories[k].insert(0, next_symbol)        
        
        for s in range(num_segments):  
            groundtruth_rates[s] += -torch.mean(torch.log(torch.clamp(subseq_ground_truth[s*segment_length:(s+1)*segment_length],min=1e-10)))
            for k in range(kn_order):
                kn_loss[k][s] += -torch.mean(torch.log(torch.clamp(kn_tks[k][s*segment_length+1:(s+1)*segment_length+1],min=1e-10)))

    result_tensor = torch.stack(kn_loss)
    norm_factor = math.floor(sequence_len/eval_bptt)
    print(f"Complete context tree {sequence_index:3d}.",flush=True)
    return (result_tensor-groundtruth_rates)/norm_factor, result_tensor/norm_factor    # regrets and loss

def compute_singleCT_unigram_loss(args):
    sequence_index, eval_sequence, ground_truth, vocab_size, eval_bptt, sequence_len, num_segments, alpha, skip = args
    
    assert eval_bptt % num_segments == 0, "eval_seq_len must be divisible by num_segments"
    segment_length = int(eval_bptt/num_segments)
    groundtruth_rates = torch.zeros(num_segments)
    # prepare for the loss vectors
    unigram_loss = torch.zeros(num_segments)
    

    assert(math.floor(sequence_len/eval_bptt)*eval_bptt < sequence_len), "sequence_len must has at least one more symbol"

    for c in range(0,math.floor((sequence_len-1)/eval_bptt)): 
        pos = c*eval_bptt        
        
        # here is a single data sequence of bptt length
        subseq = eval_sequence[pos:pos+eval_bptt+1]
        subseq_ground_truth = ground_truth[pos:pos+eval_bptt]
        loss = torch.zeros(eval_bptt+1)
        histories = torch.zeros(vocab_size)
        for j in range(eval_bptt+1):            
            next_symbol = subseq[j]
            
            if(histories[next_symbol]==0):
                p = 1/vocab_size
            else:
                p = (histories[next_symbol]+alpha)/(j+alpha*vocab_size)
            histories[next_symbol] += 1
            loss[j] = p

        for s in range(num_segments):  
            groundtruth_rates[s] += -torch.mean(torch.log(torch.clamp(subseq_ground_truth[s*segment_length:(s+1)*segment_length],min=1e-10)))
            unigram_loss[s] += -torch.mean(torch.log(torch.clamp(loss[s*segment_length+1:(s+1)*segment_length+1],min=1e-10)))
    
    norm_factor = math.floor(sequence_len/eval_bptt)
    print(f"Complete context tree {sequence_index:3d}.",flush=True)
    return (unigram_loss-groundtruth_rates)/norm_factor, unigram_loss/norm_factor    # regrets and loss

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

def kneser_ney_evaluate_mp(eval_sequences: Tensor, ground_truth: Tensor, num_workers, vocab_size: int, eval_bptt:int, sequence_len: int, number_segments: int, kn_order: int, kn_delta: float, skips):

    num_CTs = eval_sequences.size(1)

    with mp.Pool(processes=num_workers) as pool:
        # each worker is given one CT to process
        args = [(i, eval_sequences[:,i], ground_truth[:,i], vocab_size, eval_bptt, sequence_len, number_segments, kn_order, kn_delta, skips[i]) for i in range(num_CTs)]
        results = pool.map(compute_singleCT_kn_loss, args)
    
    regrets, total_loss  = zip(*results)        
    regrets_list = list(regrets)
    loss_list = list(total_loss)
    regret_tensor = torch.stack(regrets_list)
    loss_tensor = torch.stack(loss_list)

    return regret_tensor, loss_tensor

def unigram_evaluate_mp(eval_sequences: Tensor, ground_truth: Tensor, num_workers, vocab_size: int, eval_bptt:int, sequence_len: int, number_segments: int, alpha: float, skips):

    num_CTs = eval_sequences.size(1)

    with mp.Pool(processes=num_workers) as pool:
        # each worker is given one CT to process
        args = [(i, eval_sequences[:,i], ground_truth[:,i], vocab_size, eval_bptt, sequence_len, number_segments, alpha, skips[i]) for i in range(num_CTs)]
        results = pool.map(compute_singleCT_unigram_loss, args)
    
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

        pos = c*eval_bptt+D
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
        # next we need to compute the loss for each segment        
        # iterate over the segments        
        for s in range(num_segments):    
            CTW_loss[s] += torch.mean(log_p[s*segment_length:(s+1)*segment_length])        
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
    global index_count
    with torch.no_grad():        
        if model.n_layers==0:
            print('Zero-layer model does not have attention weights.')
            return
        output = model(data.unsqueeze(1))                
        # extracting attension weights        
        attention_weights = model.transformer_encoder.get_attention_weights()
        sizes = [tensor.size() for tensor in attention_weights]
        print(sizes)
        #Saving the attention weights        
        result_file = Path(os.path.join(result_dir, f"data{index_count}.mat"))
        index_count = index_count + 1
        #attention_weights_cpu = [tensor.cpu() for tensor in attention_weights]        
        attention_weights_cpu = {f'layer_{i}': tensor.detach().cpu().numpy() for i, tensor in enumerate(attention_weights)}
        data_cpu = data.cpu()
        savemat(result_file,{'attention_weights':attention_weights_cpu,'data':data_cpu})
        
        for i, layer in enumerate(model.transformer_encoder.layers):
            for j in range(layer.self_attn.num_heads):
                # Plot the attention map
                plt.figure(figsize=(10, 8))          
                attention_weights_single_layer = attention_weights[i][0,j,:,:].detach().cpu().numpy()
                plt.imshow(attention_weights_single_layer, cmap='viridis')
                plt.colorbar()
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.title(f"Attention Weights for Layer-{i+1}")
                # Save the attention map to a PDF file          
                image_file = Path(os.path.join(result_dir, f"attn_layer{i+1}_head{j+1}.pdf"))    
                plt.savefig(image_file)
                plt.close()


def eval_batch(model: nn.Module, eval_data: Tensor, ground_truth: Tensor, bptt:int, num_segments: int):
   
    criterion = nn.CrossEntropyLoss(reduction='none')  # Use 'none' to get the loss for each element
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
            subseq_ground_truth = ground_truth[pos:pos+bptt,:]

            targets = targets.view(seq_len, batch_size)

            # Compute segment indices
            segment_indices = torch.arange(num_segments).unsqueeze(1) * segment_length
            segment_indices = segment_indices + torch.arange(segment_length).unsqueeze(0)

            # Gather the output and targets for all segments
            output_segments = output[segment_indices]  # Shape: [num_segments, segment_length, batch_size, vocab_size]
            targets_segments = targets[segment_indices]  # Shape: [num_segments, segment_length, batch_size]

            # Compute total loss for all segments
            total_loss += criterion(
                output_segments.permute(2, 3, 0, 1),  # Permute to [batch_size, vocab_size, num_segments, segment_length]
                targets_segments.permute(2, 0, 1)    # Permute to [batch_size, num_segments, segment_length]
                ).mean(dim=2).cpu().numpy()  # Sum over the sequence length (dim=3)

            # Compute ground truth rates for all segments
            groundtruth_rates += -torch.mean(
                torch.log(torch.clamp(subseq_ground_truth[segment_indices], min=1e-10)),
                dim=1  # Mean over the sequence length
                ).t()

            # # iterate over the segments
            # for k in range(num_segments):
            #     output_part = output[k*segment_length:(k+1)*segment_length]                
            #     targets_part = targets[k*segment_length:(k+1)*segment_length]

            #     # Vectorized computation for all batches in the segment
            #     total_loss[:, k] += criterion(output_part.permute(1, 2, 0), targets_part.T).sum(dim=1).cpu().numpy()
            #     groundtruth_rates[:, k] += -torch.mean(torch.log(torch.clamp(subseq_ground_truth[k*segment_length:(k+1)*segment_length, :], min=1e-10)), dim=0)

            #     # for j in range(batch_size):
            #     #     output_batch = output_part[:, j, :]  # Select the j-th batch element
            #     #     targets_batch = targets_part[:, j]   # Select the j-th batch element
            #     #     total_loss[j,k] += criterion(output_batch, targets_batch).item()
            #     #     # ground_truth is already shifted by 1, so no need to shift again here
            #     #     # ground_truth is a tensor of [eval_seq_len, batch_size]
            #     #     groundtruth_rates[j,k] += -torch.mean(torch.log(torch.clamp(subseq_ground_truth[k*segment_length:(k+1)*segment_length,j],min=1e-10))) 


    norm_factor = math.floor((eval_seq_len-1)/bptt)     
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
        tfmode = cfg.tfmode


        model = CustomTransformerMixedModel(vocab_size, max_tree_depth,emsize, nheads, d_hid, nlayers, n_syn_layers ,dropout, False, tfmode).to(device)          
        # load the trained model
        model_dir = Path(cfg.modelpath)
        nheads_string = "_".join([str(item) for item in nheads])  # This joins the items with underscores
        if tfmode == 'normal':
            best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt")        
        elif tfmode == 'backward':
            # Note save to no position model directory
            best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos2.pt")        
        elif tfmode == 'totalcountonly':
            best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos1.pt")        
        elif tfmode == 'nocounts':
            best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos.pt")        
        elif tfmode == 'withoutFF':
            best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_noFF.pt")        
        elif tfmode == 'normonly':
            best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_normonly.pt")        
        else:
            raise ValueError(f"Unknown tfmode: {tfmode}. Please use 'normal', 'backward', 'totalcountonly', 'nocounts', 'withoutFF', or 'normonly'.")    
            return

        # standard model path
        #best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt")            
        # nopos model path
        #best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos2.pt")            
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
        if tfmode == 'normal':
            result_dir = os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}")        
        elif tfmode == 'backward':        
            result_dir = os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos2")        
        elif tfmode == 'totalcountonly':
            result_dir = os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos1")        
        elif tfmode == 'nocounts':
            result_dir = os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos")        
        elif tfmode == 'withoutFF':
            result_dir = os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_noFF")        
        elif tfmode == 'normonly':
            result_dir = os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_normonly")        
        else:
            raise ValueError(f"Unknown tfmode: {tfmode}. Please use 'normal', 'backward', 'totalcountonly', 'nocounts', 'withoutFF', or 'normonly'.")    
            return
        result_dir = Path(result_dir)
       
        result_dir.mkdir(parents = True, exist_ok = True)   
        regret_store = torch.zeros(num_eval_CTs,number_segments)
        loss_store = torch.zeros(num_eval_CTs,number_segments)
        # file name shows the evaluation parameter
        record_file = Path(os.path.join(result_dir, f"record_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.txt"))    
        print(f"Record file will be saved in {record_file}.")
        assert( num_eval_CTs%batch_size == 0), "num_eval_CTs must be divisible by batch_size"   

        with open(record_file, 'w') as file:
            with redirect_stdout(file): 
                print('-' * 89)                
                # evaluation loop
                for i in range(0,num_eval_CTs,batch_size):                                            
                    # We produce batch_size column from the overll data columns to train, where we 
                    # enumerate over all the num_train_contexts columns, one batch_size-column group a time                                
                    
                    # we shift the eval_sequences by the tree depth, so that for CTW, the comparison is fair
                    # note that the ground_truth is further shifted by 1
                    regrets, loss = eval_batch(model, eval_sequences[eval_max_tree_depth-1:,i:i+batch_size], ground_truth[eval_max_tree_depth:,i:i+batch_size] ,eval_bptt, number_segments)
                    # next we need store the results: should be [batch_size,number_segments]
                    #results = results - eval_rates[i:i+batch_size].unsqueeze(1) # subtract the rate, obtain the regrets
                    regret_store[i:i+batch_size] = regrets
                    loss_store[i:i+batch_size] = loss  
                
                result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))                
                #savemat(result_file,{'regret_tensor':regret_store, 'loss_tensor':loss_store,'eval_rates':eval_rates, 'eval_num_contexts':eval_num_contexts, 'eval_depths':eval_depths, 'eval_skips':eval_skips})
                savemat(result_file,{'regret_tensor':regret_store, 'loss_tensor':loss_store})
                print(f"Results saved as {result_file}.")
                for i in range(num_eval_CTs):
                    print(f"Context {i}: {regret_store[i,:]} ; {loss_store[i,:]}")
        print(f"Evaluation results saved as {result_file}.")
        
        # get attention weights and plot        
        # let's find a position with representative data
        # this model will do the weights
        model = CustomTransformerMixedModel(vocab_size, max_tree_depth, emsize, nheads, d_hid, nlayers, n_syn_layers ,dropout, True, tfmode).to(device) 
        model.load_state_dict(torch.load(best_model_params_path,weights_only=False))
        model = model.to(device)         
        CT_index = 4 
        get_attn_weights(model, eval_sequences[:eval_bptt,CT_index], result_dir)

    elif eval_mode == 'PPM':
        # sequential generation of the PPM estimate is required, this can be done using MP again
        ppm_order = cfg.ppmmodel_order
        result_dir = Path(cfg.resultpath)   
        num_workers = cfg.num_processes 
        result_dir = Path(os.path.join(result_dir, f"ppm_order{ppm_order}_{eval_bptt}"))
        result_dir.mkdir(parents = True, exist_ok = True)           
        record_file = Path(os.path.join(result_dir, f"record_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.txt"))    
        with open(record_file, 'w') as file:
            with redirect_stdout(file): 
                print('-' * 89)                
                regret_store, loss_store = ppm_evaluate_mp(eval_sequences[eval_max_tree_depth-1:,:], ground_truth[eval_max_tree_depth:,:], num_workers, vocab_size, eval_bptt, min(eval_data_seq_len,eval_data_seq_selection_len),number_segments, ppm_order, eval_skips)                                                             
                #results_store = results_store - eval_rates.unsqueeze(1).unsqueeze(1) # subtract the rate, obtain the regrets
                result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
                savemat(result_file,{'regret_tensor':regret_store, 'loss_tensor': loss_store, 'eval_rates':eval_rates, 'eval_num_contexts':eval_num_contexts, 'eval_depths':eval_depths, 'eval_skips':eval_skips})
                for i in range(num_eval_CTs):
                    print(f"Context {i}: {regret_store[i,:]} ; {loss_store[i,:]}")
        print(f"Evaluation results saved as {result_file}.")
    elif eval_mode == 'CTW': # CTW evaluation        
        p = cfg.eval_tree_nodesplit_p
        ctw_alpha = cfg.ctw_alpha
        num_workers = cfg.num_processes
        ctw_depth = cfg.ctw_depth
        # sequential generation of the PPM estimate is required, this can be done using MP again        
        result_dir = Path(cfg.resultpath) 
        # directory sets the ctw parameters   
        result_dir = Path(os.path.join(result_dir, f"ctw_p{p:.3f}_a{ctw_alpha:.2f}_{eval_bptt}"))
        #result_dir = Path(os.path.join(result_dir, f"ctw_p{p:.3f}_a{ctw_alpha:.2f}_{eval_bptt}_{ctw_depth}"))
        result_dir.mkdir(parents = True, exist_ok = True)       
        # file name shows the evaluation dataset parameter
        record_file = Path(os.path.join(result_dir, f"record_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.txt"))    
    
        with open(record_file, 'w') as file:
            with redirect_stdout(file): 
                print('-' * 89)                
                #regret_store, loss_store = ctw_evaluate_mp(eval_sequences, ground_truth, num_workers, vocab_size, eval_bptt, min(eval_data_seq_len,eval_data_seq_selection_len),number_segments, eval_max_tree_depth, 1-p, ctw_alpha, eval_skips)                                                             
                regret_store, loss_store = ctw_evaluate_mp(eval_sequences, ground_truth[eval_max_tree_depth:,:], num_workers, vocab_size, eval_bptt, min(eval_data_seq_len,eval_data_seq_selection_len),number_segments, ctw_depth, 1-p, ctw_alpha, torch.zeros(num_eval_CTs).int())                                                                             
                result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
                savemat(result_file,{'regret_tensor':regret_store, 'loss_tensor': loss_store, 'eval_rates':eval_rates, 'eval_num_contexts':eval_num_contexts, 'eval_depths':eval_depths, 'eval_skips':eval_skips})
                #savemat(result_file,{'regret_tensor':regret_store, 'loss_tensor': loss_store})
                for i in range(num_eval_CTs):
                    print(f"Context {i}: {regret_store[i,:]} ; {loss_store[i,:]}")
        print(f"Evaluation results saved as {result_file}.")
    elif eval_mode=='UG': # using Kneser-Ney smoothing
        result_dir = Path(cfg.resultpath) 
        num_workers = cfg.num_processes
        result_dir = Path(os.path.join(result_dir, f"unigram_{eval_bptt}"))
        result_dir.mkdir(parents = True, exist_ok = True)
        record_file = Path(os.path.join(result_dir, f"record_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.txt"))
        with open(record_file, 'w') as file:
            with redirect_stdout(file):
                print('-' * 89)
                regret_store, loss_store = unigram_evaluate_mp(eval_sequences[eval_max_tree_depth-1:,:], ground_truth[eval_max_tree_depth:,:], num_workers, vocab_size, eval_bptt, min(eval_data_seq_len,eval_data_seq_selection_len),number_segments, eval_alpha, eval_skips)
                result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
                savemat(result_file,{'regret_tensor':regret_store, 'loss_tensor': loss_store, 'eval_rates':eval_rates, 'eval_num_contexts':eval_num_contexts, 'eval_depths':eval_depths, 'eval_skips':eval_skips})
                for i in range(num_eval_CTs):
                    print(f"Context {i}: {regret_store[i,:]} ; {loss_store[i,:]}")
        print(f"Evaluation results saved as {result_file}.")
    else: # using Kneser-Ney smoothing
        kn_order = cfg.kn_order
        kn_delta = cfg.kn_delta
        result_dir = Path(cfg.resultpath) 
        num_workers = cfg.num_processes
        result_dir = Path(os.path.join(result_dir, f"kneser_ney_order{kn_order}_delta{kn_delta}"))
        result_dir.mkdir(parents = True, exist_ok = True)
        record_file = Path(os.path.join(result_dir, f"record_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.txt"))
        with open(record_file, 'w') as file:
            with redirect_stdout(file):
                print('-' * 89)
                regret_store, loss_store = kneser_ney_evaluate_mp(eval_sequences[eval_max_tree_depth-1:,:], ground_truth[eval_max_tree_depth:,:], num_workers, vocab_size, eval_bptt, min(eval_data_seq_len,eval_data_seq_selection_len),number_segments, kn_order, kn_delta, eval_skips)
                result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
                savemat(result_file,{'regret_tensor':regret_store, 'loss_tensor': loss_store, 'eval_rates':eval_rates, 'eval_num_contexts':eval_num_contexts, 'eval_depths':eval_depths, 'eval_skips':eval_skips})
                for i in range(num_eval_CTs):
                    print(f"Context {i}: {regret_store[i,:]} ; {loss_store[i,:]}")
        print(f"Evaluation results saved as {result_file}.")
    return
    

