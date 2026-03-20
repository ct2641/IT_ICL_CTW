import torch
from torch import nn, Tensor
import math
import os
import time
from pathlib import Path
from transformers import get_scheduler
from contextlib import redirect_stdout
import gc 

# utility for extracting data batches
from batchproc import get_batch

# simple transformer module using classical multihead attention model
from transformermodelmixed import CustomTransformerMixedModel

def val_batch(model: nn.Module, val_data: Tensor, ntokens, bptt:int, val_seq_len: int, portion = 1.0):
    assert((portion>=0) and (portion<=1))    
    criterion = nn.CrossEntropyLoss()
    model.eval()                                                # turn on train mode
    total_loss = torch.zeros(val_data.size(1))                  # Initialize total_loss as a vector of zeros with length batch_size
    
    with torch.no_grad():
        for i in range(0, val_seq_len - 1, bptt):      
            data, targets = get_batch(val_data, i, bptt)        # [bptt, batch_size], and bptt*batch_size, respectively
            output = model(data)                       
            [seq_len, batch_size] = data.size()            
            cutoff = math.floor(portion * seq_len)              # only consider the last portion to see the performance          
            output = output[-cutoff:]
            targets = targets.view(seq_len, batch_size)
            targets = targets[-cutoff:]
        
            for j in range(batch_size):
                output_batch = output[:, j, :]                  # Select the j-th batch element
                targets_batch = targets[:, j]                   
                total_loss[j] += criterion(output_batch, targets_batch).item()
            
    return total_loss / math.floor(val_seq_len/bptt) 

def train_batch(model: nn.Module, train_data, optimizer, scheduler, epoch, bch, ntokens, bptt):
    criterion = nn.CrossEntropyLoss()
    model.train()                                               # turn on train mode
    total_loss = 0.    
    start_time = time.time()    

    number_blocks = math.floor((train_data.size(0) - 1)/bptt)

    for itr in range(number_blocks):
        i = itr * bptt
        data, targets = get_batch(train_data, i, bptt)
        output = model(data)        
        output_flat = output.view(-1, ntokens)        
        loss = criterion(output_flat, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()            
        total_loss += loss.item()
    scheduler.step()
    
    num_batch = len(range(0, train_data.size(0) - 1, bptt))    
    s_per_batch = (time.time() - start_time)  
    cur_loss = total_loss / num_batch
    ppl = math.exp(cur_loss)
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    print(f'| epoch {epoch:3d} | batch {bch:2d} | s/batch {s_per_batch:5.5f} | loss {cur_loss:5.5f} | ppl {ppl:8.5f} | alloc memory {allocated_memory / 1024**2:.2f} MB | Res memory {reserved_memory / 1024**2:.2f} MB', flush=True)   
    return cur_loss 

def train_model(cfg):
    # load the CT parameters and dataset parameters
    vocab_size = cfg.vocab_size
    max_num_skip = cfg.max_num_skip
    max_tree_depth = cfg.max_tree_depth                       # the max context length is tree_depth 
    mix_depth = cfg.mix_depth
    mix_skip = cfg.mix_skip    
    num_train_CTs = cfg.num_train_CTs
    sequence_len = cfg.training_data_seq_len    
    n_syn_layers = cfg.num_synthetic_layers                   # zero = regular transformer model
    p = cfg.tree_nodesplit_p
    alpha = cfg.alpha
    tfmode = cfg.tfmode
    
    # load training paramters
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    dropout = cfg.dropout                                     # dropout probability
    early_stop_count = cfg.early_stop_count

    # load validation parameters
    num_val_CTs = cfg.num_val_CTs
    val_data_seq_len = cfg.val_data_seq_len   
    in_training_val_seq_len = cfg.in_training_val_seq_len
    
    # load transformer parameters
    bptt = cfg.bptt
    nheads = cfg.nheads                                       # number of heads in ``nn.MultiheadAttention``
    emsize = cfg.emsize 
    d_hid = cfg.d_hid                                         # dimension of the feedforward network model in ``nn.TransformerEncoder``, usually emsize or 2*emsize
    nlayers = len(nheads)                                     # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    
    # load in_training validation parameters
    num_val_tests = cfg.num_intrain_val_tests    
    val_portion = cfg.val_tail_portion
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    
    # set dataset file names
    data_dir = Path(cfg.datapath)    
    train_file_name = f"train_dataset_v{vocab_size}_d{max_tree_depth}_s{max_num_skip}_{sequence_len}_{num_train_CTs}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt"    
    train_file_path = data_dir/train_file_name    
    val_file_name = f"val_dataset_v{vocab_size}_d{max_tree_depth}_s{max_num_skip}_{val_data_seq_len }_{num_val_CTs}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt"    
    val_file_path = data_dir/val_file_name    

    # test to see if the data files already exist and then load them
    if train_file_path.exists() and val_file_path.exists():
        print(f"Train dataset exists at {train_file_path}.")
        train_data_loaded = torch.load(train_file_path,weights_only=False)    
        print(f"Training dataset loaded from {train_file_path}.")
        print(f"Val dataset exists at {val_file_path}.")
        val_data_loaded = torch.load(val_file_path,weights_only=False)
        print(f"Validation dataset loaded from {val_file_path}.")
    else:
        print(f"Training dataset not found at {train_file_path} or validation dataset not found at {val_file_path}. Need to create new dataset. Exiting.")    
        return    

    # Extract tensors from train_data
    train_sequences = train_data_loaded['data']
    
    # Extract tensors from val_data, together with other parameters
    val_sequences = val_data_loaded['data']
    val_rates = val_data_loaded['rates']
    val_num_contexts = val_data_loaded['num_contexts']
    val_depths = val_data_loaded['depths']
    val_skips = val_data_loaded['skips']

    train_sequences = train_sequences.contiguous().to(device) 
    val_sequences = val_sequences.contiguous().to(device)

    # make the model
    model = CustomTransformerMixedModel(vocab_size, max_tree_depth, emsize, nheads, d_hid, nlayers, n_syn_layers, dropout, False, tfmode).to(device)          
    
    # save the pt model file 
    model_dir = Path(cfg.modelpath)
    nheads_string = "_".join([str(item) for item in nheads])      
    if tfmode == 'normal':
        best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}.pt")        
    elif tfmode == 'backward':
        # Note save to no position model directory
        best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos2.pt")        
    elif tfmode == 'totalcountonly':
            best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos1.pt")        
    elif tfmode == 'nocounts':
        best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos.pt")        
    elif tfmode == 'withoutFF':
        best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_noFF.pt")        
    elif tfmode == 'normonly':
        best_model_params_path = os.path.join(model_dir, f"tf_best_model_params_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_normonly.pt")        
    else:
        raise ValueError(f"Unknown tfmode: {tfmode}. Please use 'normal', 'backward', 'totalcountonly', 'nocounts', 'withoutFF', or 'normonly'.")    
        return
    # save to standard model directory    
    best_model_params_path = Path(best_model_params_path)
    # load existing parameters: note must be the same network architecture
    if best_model_params_path.exists():
        print(f"Model already exists at {best_model_params_path}.")
        model.load_state_dict(torch.load(best_model_params_path,weights_only=False))
        print(f"Loaded model from {best_model_params_path}.")  
    else:
        model_dir.mkdir(parents=True, exist_ok=True)   
        
    # training setup
    best_val_diff = float('inf')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-2)
    num_training_steps = cfg.num_training_steps               # total number of training steps    
    num_warmup_steps = cfg.num_warmup_steps                   # number of warmup steps
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
        )
    no_improvement_count = 0

    # record the training results no position model 
    #model_record_dir = Path(os.path.join(model_dir, f"tf_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos2"))
    # standard training results 
    if tfmode == 'normal':
        model_record_dir = os.path.join(model_dir, f"tf_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}")        
    elif tfmode == 'backward':        
        model_record_dir = os.path.join(model_dir, f"tf_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos2")        
    elif tfmode == 'totalcountonly':
        model_record_dir = os.path.join(model_dir, f"tf_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos1")        
    elif tfmode == 'nocounts':
        model_record_dir = os.path.join(model_dir, f"tf_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_nopos")        
    elif tfmode == 'withoutFF':
        model_record_dir = os.path.join(model_dir, f"tf_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_noFF")        
    elif tfmode == 'normonly':
        model_record_dir = os.path.join(model_dir, f"tf_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}_normonly")        
    else:
        raise ValueError(f"Unknown tfmode: {tfmode}. Please use 'normal', 'wposition', 'countonly', 'nopos', or 'withoutFF'.")    
        return
    #model_record_dir = Path(os.path.join(model_dir, f"tf_sl{n_syn_layers}_{bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}"))
    model_record_dir = Path(model_record_dir)
    model_record_dir.mkdir(parents=True, exist_ok=True)  
    result_file = Path(os.path.join(model_record_dir, f"training_recording.txt"))
    # training loop
    with open(result_file, 'w') as file:
        with redirect_stdout(file): 
            for epoch in range(1, epochs + 1):
                epoch_start_time = time.time()

                print('-' * 89)
                # training
                total_loss = 0                
                for i in range(0,num_train_CTs,batch_size):                                            
                    # We produce batch_size column from the overll data columns to train, where we 
                    # enumerate over all the num_train_contexts columns, one batch_size-column group a time            
                    loss = train_batch(model, train_sequences[:,i:i+batch_size], optimizer, scheduler, epoch, int(i/batch_size+1), vocab_size, bptt)            
                    total_loss += loss                   

                print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s | total loss {total_loss:5.5f}', flush=True)

                # validation                
                val_loss_store = torch.zeros(num_val_tests)
                for i in range(0,num_val_CTs,batch_size):                                            
                    # We produce batch_size column from the overll data columns to train, where we 
                    # enumerate over all the num_train_contexts columns, one batch_size-column group a time            
                    current_val_loss = val_batch(model, val_sequences[:,i:i+batch_size], vocab_size, bptt, min(val_data_seq_len, in_training_val_seq_len), val_portion)          
                    val_loss_store[i:i+batch_size] = current_val_loss
                    
                
                val_ppl = torch.exp(val_loss_store)

                for j in range(0,num_val_tests):
                    print(f'| validation {j:2d} | CT depth {val_depths[j]:02} | #CTs {val_num_contexts[j]:02} | num-skip {val_skips[j]} | valid loss {val_loss_store[j]:2.5f} | true rate {val_rates[j]:2.5f} | regret {val_loss_store[j]-val_rates[j]:2.5f} | v-ppl {val_ppl[j]:2.5f}', flush=True)

                # training and validation time, and print summary
                elapsed = time.time() - epoch_start_time
                val_diff = (sum(val_loss_store)-sum(val_rates[:num_val_tests]))/num_val_tests               
                print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | average validation regret {val_diff:5.5f}')
                print('-' * 89, flush=True)

                # save the best model parameters and early stop if no improvement for a while
                if val_diff < best_val_diff:
                    best_val_diff = val_diff
                    torch.save(model.state_dict(), best_model_params_path)
                    no_improvement_count = 0
                else:
                    no_improvement_count = no_improvement_count + 1
                if no_improvement_count>early_stop_count:
                    break            
                
                torch.cuda.empty_cache()
                gc.collect()
                allocated_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()
                print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
                print(f"Reserved memory: {reserved_memory / 1024**2:.2f} MB", flush=True)
                    
            print(f"Training complete.")
