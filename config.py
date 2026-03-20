# config.py

# group 0-21: basic model with beta=0.85 and alpha = 0.5, max_tree_depth=3, ppmmodel_order=3
# group 60-81: basic model with beta=0.85 and alpha = 0.5, max_tree_depth=4, ppmmodel_order=4
# group 150-171: basic model with beta=0.85 and alpha = 0.5, max_tree_depth=5, ppmmodel_order=5

# group 30-51: basic model with beta=0.85 and alpha = 1.0

# group 90-111: basic model with beta=0.85 and alpha = -0.1, mixed depths = 3
# group 120-141: basic model with beta=0.965 and alpha = -0.1, mixed depths = 3 
# group 180-199: basic model with beta=0.95 and alpha = -0.1, mixed depths = 5 

# group 210-209: basic model with beta=0.965 and alpha = 0.5, max_tree_depth=5,  mixed depths = 5, mixing Dirichet
# group 240-259: basic model with beta=0.965 and alpha = 0.5, max_tree_depth=3, mixed depths = 3, mixing Dirichet


# Configuration parameters for the model and training
class BaseConfig:
    def __init__(self):
        # global model parameter: needed for both context tree and transformer model
        self.vocab_size = 3

        # data generation parameters: for training, validation, and evaluation, respectively
        # training and validation datasets should match, while the evaluation dataset can be different    
        # training and validation context tree parameters        
        self.tree_nodesplit_p = 0.85
        self.max_tree_depth = 3                   # do no allow too large tree depth
        self.max_num_skip = 0
        self.alpha = 0.5
        # training dataset parameters 
        self.mix_skip = False
        self.mix_depth = False
        self.num_train_CTs = 20000    
        self.training_data_seq_len = 5121  # can cut into training segments of bptt each: 512*10+1 for TF shift
        self.num_processes = 12
        # validation dataset parameters
        self.num_val_CTs = 512
        self.val_data_seq_len = 5121        # can cut into validation segments of bptt each: 512*10+1 for TF shift
        self.val_tail_portion = 0.25
        self.in_training_val_seq_len = 5121
        self.num_intrain_val_tests = 512     

        # evaluation context tree parameters
        self.eval_mode = 'TF'                     # 'TF', 'PPM', 'CTW', 'KN'
        self.eval_max_num_skip = 0
        self.eval_max_tree_depth = 3
        self.eval_tree_nodesplit_p = 0.85       
        self.eval_alpha = 0.5
        self.num_eval_CTs = 8192  
        # evalation dataset parameters        
        self.eval_mix_skip = False
        self.eval_mix_depth = False
        self.eval_data_seq_len = 5130             # can cut into evaluation segments of eval_bptt each, 512*10 + buffer for the late start in CTW
        self.eval_data_seq_selection_len = 5130   # the length of the sequence selected for evaluation
        self.eval_bptt = 512   
        self.number_segments = 256                # number of segments to evaluate, must divide eval_bptt
        
        # other evaluation parameters 
        self.ppmmodel_order = 3
        self.ctw_alpha = 0.5    
        self.ctw_depth = self.eval_max_tree_depth     
        self.kn_delta = 2
        self.kn_order = 3

        # transformer paramters        
        self.bptt = 512                           # context window size            
        self.num_synthetic_layers = 0             # number of synthetic tokens to add to the training data
        # different choices of transformer models: layers and number of heads parameters, after the synthetic layers        
        self.nheads = [8,8,8,8]            
        self.dropout = 0.1                        # dropout probability
        self.emsize = 64                          # embedding dimension 
        self.d_hid = 128                          # hidden dimension
        self.tfmode = 'normal'                    # 'normal', 'nocounts', 'totalcountonly', 'backward', 'withoutFF'

        self.eval_mode = 'TF'                    # 'TF', 'PPM', CTW
        # training parameters
        self.learning_rate = 0.001            
        self.early_stop_count = 20
        self.epochs = 100   
        self.num_training_steps = 10000
        self.num_warmup_steps = 10
        self.batch_size = 512                      # training batch size 
        # other in-training validation parameters
        self.eval_batch_size = 128                 # evaluation batch size

        # data storage directories
        self.datapath = 'data/LLMCTdata'
        self.modelpath = 'data/LLMCTmodels'
        self.resultpath = 'data/LLMCTresults'

class ConfigTest(BaseConfig):
    def __init__(self):
        super().__init__()
        # training dataset parameters 
        self.tree_nodesplit_p = 0.25
        self.mix_skip = False
        self.mix_depth = False
        self.num_train_CTs = 256       
        self.training_data_seq_len = 1025          # can cut into training segments of bptt each
        self.num_processes = 1
        # validation dataset parameters
        self.num_val_CTs = 256
        self.val_data_seq_len = 1025               # can cut into validation segments of bptt each
        self.val_tail_portion = 0.25
        self.in_training_val_seq_len = 1025
        self.num_intrain_val_tests = 256     

        self.eval_mode = 'PPM'                    # 'TF', 'PPM', CTW
        self.num_eval_CTs = 256  
        # evalation dataset parameters                
        self.eval_data_seq_len = 1025              # can cut into evaluation segments of eval_bptt each
        self.eval_data_seq_selection_len = 1025    # the length of the sequence selected for evaluation
        self.eval_bptt = 128   
        self.number_segments = 8                   # number of segments to evaluate, must divide eval_bptt
        
        # transformer paramters        
        self.bptt = 128                            # context window size            
        self.num_synthetic_layers = 0              # number of synthetic tokens to add to the training data
        # different choices of transformer models: layers and number of heads parameters, after the synthetic layers        
        self.nheads = [8,8]            
        
        self.batch_size = 64                       # training batch size 
        # other in-training validation parameters
        self.eval_batch_size = 64                  # validation batch size    

#######################################################################################################

class Config0(BaseConfig):
    def __init__(self):
        super().__init__()

class Config1(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config3(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config4(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
class Config5(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config6(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config7(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config8(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config9(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config10(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config11(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config12(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config13(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 64
class Config14(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 256

class Config15(BaseConfig):
    def __init__(self):
        super().__init__()
        self.d_hid = 64
class Config16(BaseConfig):
    def __init__(self):
        super().__init__()
        self.d_hid = 256

class Config17(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []

class Config1701(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.tfmode = 'nocounts'

class Config1702(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.tfmode = 'totalcountonly'

class Config1703(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.tfmode = 'backward'

class Config1704(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
        self.tfmode = 'withoutFF'

class Config1705(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,4]
        self.tfmode = 'withoutFF'

class Config1706(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,2]
        self.tfmode = 'withoutFF'

class Config1707(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,1]
        self.tfmode = 'withoutFF'

class Config1708(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,4]        

class Config1709(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,2]        

class Config1710(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,1]

class Config1711(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [1,1]

class Config1712(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [1,1,1]

class Config1713(BaseConfig):
    def __init__(self):
        super().__init__()
        self.eval_mode = 'KN'
        self.kn_delta = 0.95
        self.kn_order = 4
        self.num_synthetic_layers = 0
        self.nheads = [1,1,1]

class Config1714(BaseConfig):
    def __init__(self):
        super().__init__()
        self.eval_mode = 'UG'   

class Config1715(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [1,1,1]  
        self.tfmode = 'normonly'    


class Config18(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config19(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 256

class Config20(BaseConfig):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]



# evaluation PPM only
class Config21(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM'         
# evaluation CTW only
class Config22(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW'     
        self.num_processes = 12    

#######################################################################################################

class Config30(BaseConfig):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0
        self.eval_alpha = 1.0
        self.ctw_alpha = 1.0
class Config31(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config32(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config33(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config34(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []

class Config35(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config36(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config37(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config38(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config39(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config40(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config41(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config42(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config43(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 64
class Config44(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 256
      

class Config45(Config30):
    def __init__(self):
        super().__init__()
        self.d_hid = 64
class Config46(Config30):
    def __init__(self):
        super().__init__()
        self.d_hid = 256

class Config47(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
class Config48(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config49(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 256

class Config50(Config30):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]


# evaluation PPM only
class Config51(Config30):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM' 
# evaluation CTW only
class Config52(Config30):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW' 

#######################################################################################################

class Config60(BaseConfig):
    def __init__(self):
        super().__init__()
        self.max_tree_depth = 4
        self.ctw_depth = 4
        self.eval_max_tree_depth = 4
        self.ppmmodel_order = 4
class Config61(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config62(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config63(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config64(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []

class Config65(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config66(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config67(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config68(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config69(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config70(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config71(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config72(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config73(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 64
class Config74(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 256

class Config75(Config60):
    def __init__(self):
        super().__init__()
        self.d_hid = 64
class Config76(Config60):
    def __init__(self):
        super().__init__()
        self.d_hid = 256

class Config77(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
class Config78(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config79(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 256

class Config80(Config60):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]

# evaluation PPM only
class Config81(Config60):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM' 
# evaluation CTW only
class Config82(Config60):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW' 

#######################################################################################################

class Config90(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mix_depth = True
        self.eval_mix_depth = True
        self.eval_max_tree_depth = 3
        self.alpha = -0.1              # negative value means sparse context tree
        self.eval_alpha = -0.1
        # other in-training validation parameters
        self.eval_batch_size = 64                 # evaluation batch size
class Config91(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config92(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config93(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config94(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []

class Config95(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config96(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config97(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config98(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config99(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config100(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config101(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config102(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config103(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 64
class Config104(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 256

class Config105(Config90):
    def __init__(self):
        super().__init__()
        self.d_hid = 64
class Config106(Config90):
    def __init__(self):
        super().__init__()
        self.d_hid = 256

class Config107(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
class Config108(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config109(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 256

class Config110(Config90):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]

# evaluation PPM only
class Config111(Config90):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM' 
# evaluation CTW only
class Config112(Config90):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW' 
        self.ctw_alpha = 0.5         

#######################################################################################################

class Config120(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mix_depth = True
        self.eval_mix_depth = False
        self.eval_max_tree_depth = 3
        self.alpha = -0.1              # negative value means sparse context tree
        self.eval_alpha = -0.1
        self.tree_nodesplit_p = 0.965
        self.eval_tree_nodesplit_p = 0.965        

class Config121(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config122(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config123(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config124(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []

class Config125(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config126(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config127(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config128(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config129(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config130(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config131(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config132(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config133(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 64
class Config134(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 256

class Config135(Config120):
    def __init__(self):
        super().__init__()
        self.d_hid = 64
class Config136(Config120):
    def __init__(self):
        super().__init__()
        self.d_hid = 256

class Config137(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
class Config138(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config139(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 256

class Config140(Config120):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]

# evaluation PPM only
class Config141(Config120):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM' 
# evaluation CTW only
class Config142(Config120):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW' 
        self.ctw_alpha = 0.5         

#######################################################################################################

class Config150(BaseConfig):
    def __init__(self):
        super().__init__()
        self.max_tree_depth = 5
        self.eval_max_tree_depth = 5
        self.ctw_depth = 5
        self.ppmmodel_order = 5
        self.bptt = 1536
        self.eval_bptt = 1536
        self.emsize = 128
        self.d_hid = 256
        self.batch_size = 128
        self.eval_batch_size = 32
        self.num_eval_CTs = 8192
        self.number_segments = 768                  # number of segments to evaluate, must divide eval_bptt1

class Config151(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config152(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config153(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config154(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []

class Config155(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config156(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config157(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config158(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config159(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config160(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config161(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config162(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config163(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 128
class Config164(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 512

class Config165(Config150):
    def __init__(self):
        super().__init__()
        self.d_hid = 128
class Config166(Config150):
    def __init__(self):
        super().__init__()
        self.d_hid = 512

class Config167(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
class Config168(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config169(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 512

class Config170(Config150):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]

# evaluation PPM only
class Config171(Config150):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM'         

# evaluation CTW only
class Config172(Config150):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW' 

# evaluation KN only
class Config173(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.8
        self.kn_order = 5
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config174(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.95
        self.kn_order = 5
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config175(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.8
        self.kn_order = 4
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config176(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.95
        self.kn_order = 4
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config177(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.8
        self.kn_order = 3
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config178(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.95
        self.kn_order = 3
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config179(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.8
        self.kn_order = 2
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config17900(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.95
        self.kn_order = 2
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config17901(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.8
        self.kn_order = 1
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config17902(Config150):
    def __init__(self):
        super().__init__()    
        self.kn_delta = 0.95
        self.kn_order = 1
        self.eval_mode = 'KN' 
        self.num_processes = 10

class Config17903(Config150):
    def __init__(self):
        super().__init__()
        self.eval_mode = 'UG'      

# class Config1770(Config150):
#     def __init__(self):
#         super().__init__()    
#         self.eval_mode = 'TF' 
#         self.tfmode = 'withoutFF'        
#         self.num_synthetic_layers = 0
#         self.nheads = [8,8]







#######################################################################################################

class Config180(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mix_depth = True
        self.eval_mix_depth = False
        self.max_tree_depth = 5
        self.eval_max_tree_depth = 5
        self.ctw_depth = 5
        self.tree_nodesplit_p = 0.95
        self.eval_tree_nodesplit_p = 0.95
        self.alpha = -0.1              # negative value means sparse context tree
        self.eval_alpha = -0.1
        self.ppmmodel_order = 5
        self.bptt = 1536
        self.eval_bptt = 1536
        self.emsize = 128
        self.d_hid = 256
        self.batch_size = 128
        self.eval_batch_size = 32
class Config181(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config182(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config183(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config184(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []

class Config185(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config186(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config187(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config188(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config189(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config190(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config191(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config192(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config193(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
class Config194(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config195(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 512
class Config196(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 512

class Config197(Config180):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]

# evaluation PPM only
class Config198(Config180):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM' 
# evaluation CTW only
class Config199(Config180):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW' 
        #self.eval_mix_depth = True 
        #self.num_processes = 4 
        self.ctw_alpha = 0.5   
          

#######################################################################################################

class Config210(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mix_depth = True
        self.eval_mix_depth = False
        self.max_tree_depth = 5
        self.eval_max_tree_depth = 5
        self.ctw_depth = 5
        #self.eval_max_tree_depth = 3        
        self.tree_nodesplit_p = 0.965
        self.eval_tree_nodesplit_p = 0.965
        self.alpha = 0.5              # negative value means sparse context tree
        
        self.eval_alpha = 0.5
        self.ppmmodel_order = 5
        self.bptt = 1536
        self.eval_bptt = 1536
        self.emsize = 128
        self.d_hid = 256
        self.batch_size = 128
        self.eval_batch_size = 32
class Config211(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config212(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config213(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config214(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []

class Config215(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config216(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config217(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config218(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config219(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config220(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config221(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config222(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config223(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
class Config224(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config225(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 512
class Config226(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 512

class Config227(Config210):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]

# evaluation PPM only
class Config228(Config210):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM' 
# evaluation CTW only
class Config229(Config210):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW' 

 #################################
 ###only for special test#########
 #################################
class Config230(Config210):
    def __init__(self):
        super().__init__()  
        self.eval_max_tree_depth = 3
        self.ctw_depth = 5     # the default is eval_max_tree_depth, but here make it different for testing purpose
        self.eval_mode = 'CTW'  


#######################################################################################################

class Config240(BaseConfig):
    def __init__(self):
        super().__init__()
        self.mix_depth = True
        self.eval_mix_depth = False        
        self.tree_nodesplit_p = 0.965
        self.eval_tree_nodesplit_p = 0.965
        
class Config241(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8]
class Config242(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8]
class Config243(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8]
class Config244(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []

class Config245(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,4,4]
class Config246(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8,8,8,8]
class Config247(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = [8,8,8]
class Config248(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 3
        self.nheads = [8,8]
class Config249(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = [8]

class Config250(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8]
class Config251(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8]
class Config252(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8,8,8,8,4,4]

class Config253(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
class Config254(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 1
        self.nheads = [8]
class Config255(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 2
        self.nheads = []
        self.d_hid = 256
class Config256(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 4
        self.nheads = []
        self.d_hid = 256

class Config257(Config240):
    def __init__(self):
        super().__init__()
        self.num_synthetic_layers = 0
        self.nheads = [8]

# evaluation PPM only
class Config258(Config240):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'PPM' 
# evaluation CTW only
class Config259(Config240):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW' 
        
#######################################################################################################

# one layer transformer with a lot of heads

class Config270(BaseConfig):
    def __init__(self):
        super().__init__()
        # data generation parameters: for training, validation, and evaluation, respectively
        # training and validation datasets should match, while the evaluation dataset can be different    
        # training and validation context tree parameters                
        self.max_tree_depth = 2                   # do no allow too large tree depth                        
        # evaluation context tree parameters
        self.eval_max_tree_depth = 2
        # evalation dataset parameters                
        
        # other evaluation parameters 
        self.ppmmodel_order = 2
        self.ctw_alpha = 0.5    
        self.ctw_depth = self.eval_max_tree_depth     

        # transformer paramters        
        self.nheads = [16]                    
        self.eval_mode = 'TF'                    # 'TF', 'PPM', CTW
        # training parameters
        

class Config271(Config270):
    def __init__(self):
        super().__init__()        
        self.nheads = [16,16]

class Config272(Config270):
    def __init__(self):
        super().__init__()        
        self.nheads = [8,8,8,8]


class Config273(Config270):
    def __init__(self):
        super().__init__()        
        self.eval_mode = 'CTW'                    # 'TF', 'PPM', CTW



#######################################################################################################

class Config280(BaseConfig):
    def __init__(self):
        super().__init__()
        self.nheads = [1,1]
class Config281(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2]
class Config282(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1]
class Config283(BaseConfig):
    def __init__(self):
        super().__init__()
        self.nheads = [2,2,2]        
class Config284(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1,1]
class Config285(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2,2,2]

class Config286(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1]        
        self.emsize = 16                          
        self.d_hid = 32                          
class Config287(Config286):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2]
class Config288(Config286):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1]
class Config289(Config286):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2,2]
class Config290(Config286):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1,1]
class Config291(Config286):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2,2,2]

class Config292(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1]        
        self.emsize = 8                         
        self.d_hid = 16                          
class Config293(Config292):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2]
class Config294(Config292):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1]
class Config295(Config292):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2,2]
class Config296(Config292):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1,1]
class Config297(Config292):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2,2,2]


class Config298(BaseConfig):
    def __init__(self):
        super().__init__()
        self.nheads = [1,1]
        self.dropout = 0.0                       # dropout probability
class Config299(Config298):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2]
class Config300(Config298):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1]
class Config301(Config298):
    def __init__(self):
        super().__init__()
        self.nheads = [2,2,2]        
class Config302(Config298):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1,1]
class Config303(Config298):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2,2,2]


class Config304(BaseConfig):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1]        
        self.emsize = 12                         
        self.d_hid = 24                          
class Config305(Config304):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2]
class Config306(Config304):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1]
class Config307(Config304):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2,2]
class Config308(Config304):
    def __init__(self):
        super().__init__()        
        self.nheads = [1,1,1,1]
class Config309(Config304):
    def __init__(self):
        super().__init__()        
        self.nheads = [2,2,2,2]
