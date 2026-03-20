import scipy.io
import matplotlib.pyplot as plt
import os
from pathlib import Path
import config
import math
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.patches as patches

# Change the font of the plot
import matplotlib
matplotlib.rc('font', size=14)          # controls default text sizes
matplotlib.rc('axes', titlesize=14)     # Font size for axes titles
matplotlib.rc('axes', labelsize=18)     # Font size for x and y labels
matplotlib.rc('xtick', labelsize=14)    # Font size for x tick labels
matplotlib.rc('ytick', labelsize=15)    # Font size for y tick labels
matplotlib.rc('legend', fontsize=14)    # Font size for legend
matplotlib.rc('figure', titlesize=16)   # Font size for figure title
plt.figure(2, figsize=(9, 7))



# Plotting parameters
line_styles = ['-', ':', '--', '-.' ]  # Solid, dashed, dash-dot, dotted
markers = ['o', 's', '^', '<', '>','x','+','v','.','p','H']  # Circle, square, diamond, triangle
colors = ['b', 'g', 'r', 'c', 'k' , 'aquamarine']  # Blue, green, red, cyan, black, magenta, yellow

counter = 0
index_list = [170, 160, 161, 150, 155,162, 174,17903,172]

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
        tfmode = cfg.tfmode        

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
        else:
            raise ValueError(f"Unknown tfmode: {tfmode}. Please use 'normal', 'wposition', 'countonly', 'nopos', or 'withoutFF'.")    
      
        result_dir = Path(result_dir)
        result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
        if result_file.exists():
            data = scipy.io.loadmat(result_file)
            x = data['regret_tensor']
            plt.figure(1)
            plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), x.mean(axis=0), 
                    linestyle=line_styles[ int(d_hid/64) % len(line_styles)], 
                    marker=markers[counter%len(markers)], 
                    markevery=math.floor(trained_bptt/20),  
                    color=colors[(n_syn_layers+nlayers) % len(colors)],                         
                    label=f'Transformer-{nlayers} layers',linewidth=1.0)           
            y = data['loss_tensor']
            plt.figure(2)
            plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), y.mean(axis=0), 
                    linestyle=line_styles[ int(d_hid/64) % len(line_styles)], 
                    marker=markers[counter%len(markers)], 
                    markevery=math.floor(trained_bptt/20),  
                    color=colors[(n_syn_layers+nlayers) % len(colors)],                         
                    label=f'Transformer-{nlayers} layers',linewidth=1.0)
            counter += 1            
        else:
            print(f"---------> Config{index}: File {result_file} does not exist.")
        
        
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
            y = data['regret_tensor']
            z = data['loss_tensor']
            for j in [2,4]:#range(ppm_order):            
                plt.figure(1)
                x = y[:,j,:]                 
                plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), x.mean(axis=0), 
                            linestyle=line_styles[1], 
                            marker=markers[counter%len(markers)], 
                            markevery=math.floor(trained_bptt/20),  
                            color=colors[j], 
                            label=f'PPM order {j+1}',linewidth=1.0)
                plt.figure(2)
                x = z[:,j,:] 
                plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), x.mean(axis=0), 
                            linestyle=line_styles[1], 
                            marker=markers[counter%len(markers)], 
                            markevery=math.floor(trained_bptt/20),  
                            color=colors[j], 
                            label=f'PPM order {j+1}',linewidth=1.0)
                counter += 1
        else:
            print(f"--------->Config{index}: File {result_file} does not exist")

    elif eval_mode == 'CTW': # CTW evaluation
        
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
            x = data['regret_tensor']
            plt.figure(1)
            plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), x.mean(axis=0), 
                        linestyle='-.', 
                        marker='*', 
                        markevery=math.floor(trained_bptt/20),  
                        color='m',                             
                        label='CTW',linewidth=1.0)           
            y = data['loss_tensor']
            plt.figure(2)
            plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), y.mean(axis=0),  
                        linestyle='-.', 
                        marker='*', 
                        markevery=math.floor(trained_bptt/20),  
                        color='m',                             
                        label='CTW',linewidth=1.0)                   
        else: 
            print(f"--------->Config{index}: File {result_file} does not exist")
    elif eval_mode == 'UG': # unit gram
        result_dir = Path(cfg.resultpath)    
       
        result_dir = Path(os.path.join(result_dir, f"unigram_{eval_bptt}"))               
        result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{p:.3f}_a{alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))
        if result_file.exists():
            data = scipy.io.loadmat(result_file)
            x = data['regret_tensor']
            plt.figure(1)
            plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), x.mean(axis=0), 
                        linestyle='-.', 
                        marker='s', 
                        markevery=math.floor(trained_bptt/20),  
                        color='m',                             
                        label='Unigram',linewidth=1.0)           
            y = data['loss_tensor']
            plt.figure(2)
            plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), y.mean(axis=0),  
                        linestyle='-.', 
                        marker='s', 
                        markevery=math.floor(trained_bptt/20),  
                        color='m',                             
                        label='Unigram',linewidth=1.0)                   
        else: 
            print(f"--------->Config{index}: File {result_file} does not exist")
    else: # KN   
        kn_order = cfg.kn_order
        kn_delta = cfg.kn_delta
        
        result_dir = Path(os.path.join(result_dir, f"kneser_ney_order{kn_order}_delta{kn_delta}"))
        result_dir.mkdir(parents = True, exist_ok = True)
        result_file = Path(os.path.join(result_dir, f"loss_v{vocab_size}_d{eval_max_tree_depth}_s{eval_max_num_skip}_{eval_data_seq_len}_{num_eval_CTs}_p{eval_p:.3f}_a{eval_alpha:.2f}_{'mixd' if eval_mix_depth==True else 'singled'}{'mixs' if eval_mix_skip==True else 'singles'}.mat"))        
        if result_file.exists():
            data = scipy.io.loadmat(result_file)
            y = data['regret_tensor']
            z = data['loss_tensor']
            #for j in range(kn_order):            
            #j = kn_order-1 # only plot the highest order 
            
            #for j in [2,4]:            
            for j in range(kn_order):            
                x = y[:,j,:] 
                plt.figure(1)
                #plt.errorbar(range(x.shape[1]), x.mean(axis=0), yerr=x.std(axis=0), linestyle=line_styles[0], marker=markers[j], color=colors[j], capsize=5, label=f'PPM order {j+1}')
                plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), x.mean(axis=0), 
                            linestyle='-.', 
                            marker=markers[counter%len(markers)], 
                            markevery=math.floor(trained_bptt/20),  
                            color=colors[j+1], 
                            label=f'KN order {j+1}', linewidth=1.0)
                plt.figure(2)
                x = z[:,j,:] 
                plt.plot(range(1,trained_bptt+1,math.floor(trained_bptt/x.shape[1])), x.mean(axis=0), 
                            linestyle='-.', 
                            marker=markers[counter%len(markers)], 
                            markevery=math.floor(trained_bptt/20),  
                            color=colors[j+1], 
                            label=f'KN order {j+1}',linewidth=1.0)
                counter += 1
        else:
            print(f"--------->Config{index}: File {result_file} does not exist")

plt.figure(1)
plt.xlabel('position in the context window')
plt.ylabel('rate regrets')
plt.legend()
plt.xlim(1, trained_bptt)
plt.title(f'Regret: alpha {"{:.2f}".format(alpha) if alpha > 0 else "mixed"} and context tree depth {eval_max_tree_depth}')

plt.figure(2)
plt.xlabel('position in the context window')
plt.ylabel('rates')
plt.legend()
plt.title(r'Compression rates: $\alpha=$'+f'{"{:.2f}".format(alpha) if alpha > 0 else "mixed"} and ' + r'$D=$' + f'{eval_max_tree_depth}')
plt.xlim(0.8, trained_bptt+0.2)


plt.savefig(f'ICL_tfdepth5_{eval_max_tree_depth}_withKN.pdf')
plt.show()
