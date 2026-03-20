import scipy.io
import matplotlib.pyplot as plt
import os
from pathlib import Path
import config
import argparse
import numpy as np


index_list = [20, 10, 11, 0, 5, 12, 22, 21]
#index_list = [80, 70, 71, 60, 65, 72, 82, 81]
#index_list = [170, 160, 161, 150, 155, 162, 172, 171]


PPM_rates_vec = np.empty(5)
rates_vec = np.empty(len(index_list)-1)

for index in index_list:

    config_class = getattr(config, f'Config{index}')
    cfg = config_class()
    result_dir = Path(cfg.resultpath)    
    p = cfg.tree_nodesplit_p
    alpha = cfg.alpha

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

    if eval_mode == 'TF':
        trained_bptt = cfg.bptt                
        nheads = cfg.nheads  
        emsize = cfg.emsize    
        p = cfg.tree_nodesplit_p
        alpha = cfg.eval_alpha    
        nlayers = len(nheads)    
        n_syn_layers = cfg.num_synthetic_layers
        dropout = cfg.dropout     
        max_num_skip = cfg.max_num_skip
        max_tree_depth = cfg.max_tree_depth  # the max context length is  (tree_depth - 1)            
        mix_skip = cfg.mix_skip
        mix_depth = cfg.mix_depth
        batch_size = cfg.eval_batch_size
        d_hid = cfg.d_hid
        nheads_string = "_".join([str(item) for item in nheads])  # This joins the items with underscores
        result_dir = Path(cfg.resultpath)  

        result_dir = Path(os.path.join(result_dir, f"tf_sl{n_syn_layers}_{trained_bptt}_v{vocab_size}_h{nheads_string}_eb{emsize}_dh{d_hid}_d{max_tree_depth}_s{max_num_skip}_p{p:.3f}_a{alpha:.2f}_{'mixd' if mix_depth==True else 'singled'}{'mixs' if mix_skip==True else 'singles'}"))
        result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
        if result_file.exists():
            data = scipy.io.loadmat(result_file)
            x = data['loss_tensor']            
        else:
            print(f"---------> Config{index}: File {result_file} does not exist.")

        rates_vec[index_list.index(index)] = x.mean()
        
        
    elif eval_mode == 'PPM':
        #continue
        # sequential generation of the PPM estimate is required, this can be done using MP again
        ppm_order = cfg.ppmmodel_order
        result_dir = Path(cfg.resultpath)   
        num_workers = cfg.num_processes 
        result_dir = Path(os.path.join(result_dir, f"ppm_order{ppm_order}_{eval_bptt}"))        
        result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
        if result_file.exists():
            data = scipy.io.loadmat(result_file)            
            z = data['loss_tensor']
            for j in range(ppm_order):                            
                x = z[:,j,:]                 
                PPM_rates_vec[j] = x.mean()
        else:
            print(f"--------->Config{index}: File {result_file} does not exist")

    else: # CTW evaluation
        
        p = cfg.eval_tree_nodesplit_p
        ctw_alpha = cfg.ctw_alpha
        ctw_depth = cfg.ctw_depth
        
        result_dir = Path(cfg.resultpath)    
        if index== 232:
            result_dir = Path(os.path.join(result_dir, f"ctw_p{p:.3f}_a{ctw_alpha:.2f}_{eval_bptt}_{ctw_depth}"))
        else:
            result_dir = Path(os.path.join(result_dir, f"ctw_p{p:.3f}_a{ctw_alpha:.2f}_{eval_bptt}"))
        result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{p:.3f}_a{alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
        if result_file.exists():
            data = scipy.io.loadmat(result_file)
            x = data['loss_tensor']            
            rates_vec[index_list.index(index)] = x.mean()
        else:
            print(f"--------->Config{index}: File {result_file} does not exist")


print(rates_vec)
print(PPM_rates_vec)


