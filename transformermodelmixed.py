import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class CustomEmbedding(nn.Module):
    def __init__(self, ntoken: int, max_prefix_len: int, d_model: int, n_syn_layers:int, 
                 dropout: float = 0.1, tfmode: str = 'normal'):
        # note we are testing the model with backward statistics when backward is True
        # for the standard model, we set backward to False
        super().__init__()
        self.ntoken = ntoken
        self.d_model = d_model
        self.n_syn_layers = n_syn_layers
        self.max_prefix_len = max_prefix_len        
        self.tfmode = tfmode
        self.backward = False
        if tfmode == 'bacward':
            self.backward = True
            assert n_syn_layers == 2, "Only backward mode is supported for 2 synthetic layers"

        # note: no dropout since nothing is trainable here

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
        Returns:
            embedded Tensor of shape ``[seq_len, batch_size, d_model]``
        """
        # Implement the custom embedding logic below
        # this should not happen, but just in case      
        if self.n_syn_layers == 0:
            return src
                
        seq_len, batch_size = src.size()        
        # Create an extended one-hot tensor of d_model, with the first several positions used as the one-hot embedding
        extended_one_hot = torch.zeros(seq_len, batch_size, self.d_model, device=src.device)
        
        # Use advanced indexing to set the appropriate positions to 1
        max_tree_depth =  self.max_prefix_len                                  # the max context length is tree_depth, hardcoded here for now  
        extended_one_hot.scatter_(2, src.unsqueeze(-1), 1)
        for i in range(1,max_tree_depth+1):
            src_temp = torch.zeros_like(src) 
            src_temp[i:,:] = src[:-i,:] 
            src_temp = src_temp + i*self.ntoken             # this delay the src sequence time-wise by i, and then change the values such that
            extended_one_hot.scatter_(2, src_temp.unsqueeze(-1), 1)
            extended_one_hot[0:i,:,self.ntoken*i] = 0 
        # This constructed layer 1: i.e., build the context
        if self.n_syn_layers == 1:
            return extended_one_hot
        ##############################################################################################################################
        ##############################################################################################################################
        
        if self.n_syn_layers == 2 and self.backward:
            # this is the backward version of the embedding, which is not used in the current construction. It can recover all counts
            mask = torch.tril(torch.ones(seq_len, seq_len, device=extended_one_hot.device), diagonal=-1).bool()
            j_ntoken_part = extended_one_hot[:, :, :self.ntoken]  # Shape: [seq_len, batch_size, self.ntoken]
            expanded_j_ntoken_part = j_ntoken_part.unsqueeze(0).expand(seq_len, -1, -1, -1)  # Shape: [seq_len, seq_len, batch_size, self.ntoken]
            # for i in range(1,max_tree_depth+1): # this build the 1st, 2nd, and 3rd order statistics, together with the counts
            for i in range(max_tree_depth,0,-1):
                j_ntoken_part_backward = extended_one_hot[:, :, self.ntoken*(i+1):self.ntoken*(i+2)]  # Shape: [seq_len, batch_size, self.ntoken]
                expanded_j_ntoken_part_backward = j_ntoken_part_backward.unsqueeze(0).expand(seq_len, -1, -1, -1)  # Shape: [seq_len, seq_len, batch_size, self.ntoken]

                # Step 1: extract vectors to find matching positions
                i_part = extended_one_hot[:, :, :self.ntoken*i]
                j_part = extended_one_hot[:, :, self.ntoken:self.ntoken*(i+1)]
                # Create a 4D tensor by expanding and comparing across the sequence: 1 if matches, 0 if not
                matches = (j_part.unsqueeze(0) == i_part.unsqueeze(1)).all(dim=-1).float()

                # Step 2: Mask the upper triangular part to only consider positions j < i
                masked_matches = matches.masked_fill(~mask.unsqueeze(-1), 0)  # Shape: [seq_len, seq_len, batch_size]

                # Step 3: Expand the mask to apply it to the first self.ntoken dimensions
                masked_matches_expanded = masked_matches.unsqueeze(-1).expand(-1, -1, -1, self.ntoken)

                # Step 5: Use masked_matches_expanded to index and sum the matched j positions
                # Multiply the mask with the j_ntoken_part to zero out non-matching elements
                masked_j_ntoken_part = expanded_j_ntoken_part * masked_matches_expanded  # Shape: [seq_len, seq_len, batch_size, self.ntoken]
                masked_j_ntoken_part_backward = expanded_j_ntoken_part_backward * masked_matches_expanded  # Shape: [seq_len, seq_len, batch_size, self.ntoken]

                # Sum along the j dimension (dim=1) to get the sum of matched values
                sum_ntoken_values = masked_j_ntoken_part.sum(dim=1)  # Shape: [seq_len, batch_size, self.ntoken]
                sum_ntoken_values_backward = masked_j_ntoken_part_backward.sum(dim=1)  # Shape: [seq_len, batch_size, self.ntoken]

                # Step 6: Calculate the average by dividing the sum by the number of matches
                # Count the number of matches, ensuring to avoid division by zero
                match_counts = masked_matches.sum(dim=1).unsqueeze(-1) # Shape: [seq_len, batch_size, 1]
                average_ntoken_values = sum_ntoken_values / match_counts.expand(-1, -1, self.ntoken).clamp(min=1)
                average_ntoken_values_backward = sum_ntoken_values_backward / match_counts.expand(-1, -1, self.ntoken).clamp(min=1)

                # Step 7: Concatenate the average values in the original tensor along the v_dim dimension
                pos = self.ntoken*(max_tree_depth+i)
                extended_one_hot[:, :, pos: pos + self.ntoken] = average_ntoken_values
                pos = self.ntoken*(max_tree_depth*2+i)
                extended_one_hot[:, :, pos: pos + self.ntoken] = average_ntoken_values_backward

            # Needs to process uni-gram separately because the matching will yield an empty tensor
            # The mask is simple this time
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, batch_size, self.ntoken)
            masked_j_ntoken_part = expanded_j_ntoken_part * mask_expanded  # Shape: [seq_len, seq_len, batch_size, self.ntoken]        
            sum_ntoken_values = masked_j_ntoken_part.sum(dim=1)  # Shape: [seq_len, batch_size, self.ntoken]
            match_counts =  mask_expanded[:,:,:,0:1].sum(dim=1) # Shape: [seq_len, batch_size, 1]
            average_ntoken_values = sum_ntoken_values / match_counts.expand(-1, -1, self.ntoken).clamp(min=1)
            pos = self.ntoken*(3*max_tree_depth+1)
            extended_one_hot[:, :, pos: pos + self.ntoken] = average_ntoken_values

            # create (1, cos(pos*pi/N), sin(pos*pi/N)) positional encoding
            # insert the positional encoding at the end of the embedding
            seq_len, batch_size, d_model = extended_one_hot.shape
            pos = torch.arange(0, seq_len, device=extended_one_hot.device).unsqueeze(1).expand(-1, batch_size)
            extended_one_hot[:, :, -3] = 1
            extended_one_hot[:, :, -2] = torch.cos(pos * math.pi / (2 * seq_len) )
            extended_one_hot[:, :, -1] = torch.sin(pos * math.pi / (2 * seq_len) )

            return extended_one_hot

        else:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=extended_one_hot.device), diagonal=-1).bool()
            # Extract the first self.ntoken dimensions for all positions for the counting
            j_ntoken_part = extended_one_hot[:, :, :self.ntoken]  # Shape: [seq_len, batch_size, self.ntoken]        
            expanded_j_ntoken_part = j_ntoken_part.unsqueeze(0).expand(seq_len, -1, -1, -1)  # Shape: [seq_len, seq_len, batch_size, self.ntoken]
            for i in range(1,max_tree_depth+1): # this build the 1st, 2nd, and 3rd order statistics, together with the counts
                
                # Step 1: extract vectors to find matching positions
                # this should count the full matched positions                    
                i_part = extended_one_hot[:, :, :self.ntoken*i]
                j_part = extended_one_hot[:, :, self.ntoken:self.ntoken*(i+1)]
                # Create a 4D tensor by expanding and comparing across the sequence: 1 if matches, 0 if not
                matches = (j_part.unsqueeze(0) == i_part.unsqueeze(1)).all(dim=-1).float()

                # Step 2: Mask the upper triangular part to only consider positions j < i            
                masked_matches = matches.masked_fill(~mask.unsqueeze(-1), 0)  # Shape: [seq_len, seq_len, batch_size]

                # Step 3: Expand the mask to apply it to the first self.ntoken dimensions
                masked_matches_expanded = masked_matches.unsqueeze(-1).expand(-1, -1, -1, self.ntoken)
                
                # Step 5: Use masked_matches_expanded to index and sum the matched j positions
                # Multiply the mask with the j_ntoken_part to zero out non-matching elements
                masked_j_ntoken_part = expanded_j_ntoken_part * masked_matches_expanded  # Shape: [seq_len, seq_len, batch_size, self.ntoken]

                # Sum along the j dimension (dim=1) to get the sum of matched values
                sum_ntoken_values = masked_j_ntoken_part.sum(dim=1)  # Shape: [seq_len, batch_size, self.ntoken]

                # Step 6: Calculate the average by dividing the sum by the number of matches
                # Count the number of matches, ensuring to avoid division by zero
                match_counts = masked_matches.sum(dim=1).unsqueeze(-1) # Shape: [seq_len, batch_size, 1]
                average_ntoken_values = sum_ntoken_values / match_counts.expand(-1, -1, self.ntoken).clamp(min=1)

                # Step 7: Concatenate the average values in the original tensor along the v_dim dimension
                pos = self.ntoken*(max_tree_depth+i)
                extended_one_hot[:, :, pos: pos + self.ntoken] = average_ntoken_values
                # Note: We canget rid of these counts and see if they can be learned naturally with one additional layer of transformers
                # as suggested by the exploration

                ###########################################################################################
                # let's try to remove the counts and see if the transformer can learn the counts
                # temperary disable # remember to enable after this test
                if self.tfmode != 'totalcountonly' and self.tfmode != 'nocounts':
                    pos = self.ntoken*(max_tree_depth*2+2)+i-1
                    extended_one_hot[:, :, pos: pos+1] = match_counts   
                ###########################################################################################

            # Needs to process uni-gram separately because the matching will yield an empty tensor
            # The mask is simple this time
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, batch_size, self.ntoken)
            masked_j_ntoken_part = expanded_j_ntoken_part * mask_expanded  # Shape: [seq_len, seq_len, batch_size, self.ntoken]        
            sum_ntoken_values = masked_j_ntoken_part.sum(dim=1)  # Shape: [seq_len, batch_size, self.ntoken]
            match_counts =  mask_expanded[:,:,:,0:1].sum(dim=1) # Shape: [seq_len, batch_size, 1]
            average_ntoken_values = sum_ntoken_values / match_counts.expand(-1, -1, self.ntoken).clamp(min=1)
            pos = self.ntoken*(max_tree_depth+max_tree_depth+1)
            extended_one_hot[:, :, pos: pos + self.ntoken] = average_ntoken_values        
            
            
            # what if we don't include the total counts, i.e., the absolute positional encoding?
            ###########################################################################################
            # let's try to remove the counts and see if the transformer can learn the counts
            # temperary disable # remember to enable after this test
            if self.tfmode != 'nocounts':
                pos = self.ntoken*(max_tree_depth*2+2)+max_tree_depth
                extended_one_hot[:, :, pos:pos+1] = match_counts 
            ###########################################################################################

        # This constructed layer 2: calculating the statistical vector and also the cumulative counts
        if self.n_syn_layers == 2:
            return extended_one_hot
        ##############################################################################################################################
        ##############################################################################################################################


        # first delay & copy over the statistical vector and counts
        pos1 = self.ntoken*(max_tree_depth*2+2) + max_tree_depth + 1         # end of the current packing position in the embedding space
        pos2 = self.ntoken*(max_tree_depth+1)                                # start of the statistical vectors and counts in the embedding space
        pos3= pos1+(self.ntoken+1)*(max_tree_depth+1)                        # end of the position in the embedding space after copying/appending (actually the length)    
        extended_one_hot[1:,:,pos1:pos3] = extended_one_hot[:-1,:,pos2:pos1] # append the previous-time vector to the current

        # next we can compute the cross-entropy loss for each depth
        # no need to separately process uni-gram since the statistical vectors have been collected
        for i in range(0,max_tree_depth+1):
            pos = pos1 + i*self.ntoken
            probabilities = extended_one_hot[:,:,pos:pos+self.ntoken]
            log_probabilities = torch.log(probabilities.clamp(min=1e-9))  # Shape: [seq_len, batch_size, 2]
            nll_loss = F.nll_loss(log_probabilities.permute(0, 2, 1), src, reduction='none')  # Shape: [seq_len, batch_size]
            pos = pos3+i
            extended_one_hot[:, :, pos:pos+1] = nll_loss.unsqueeze(-1)
        # This constructed layer 3: calculate the individual cross-entropy loss
        # Now the embedding vector grows to pos3+max_tree_depth+1
        if self.n_syn_layers == 3:
            return extended_one_hot
        ##############################################################################################################################
        ##############################################################################################################################
         
        for i in range(1,max_tree_depth+1): 
            # these selective sums are identical to the previous layer: we can combine them to make it more efficient but for 
            # clarity reasons, we repeat here (to remove this layer when necessary)

            # Step 1: extract vectors to find matching positions 
            # this should count the full matched positions                    
            i_part = extended_one_hot[:, :, :self.ntoken*i]
            j_part = extended_one_hot[:, :, self.ntoken:self.ntoken*(i+1)]
            # Create a 4D tensor by expanding and comparing across the sequence: 1 if matches, 0 if not
            matches = (j_part.unsqueeze(0) == i_part.unsqueeze(1)).all(dim=-1).float()
            # Step 2: Mask the upper triangular part to only consider positions j < i            
            masked_matches = matches.masked_fill(~mask.unsqueeze(-1), 0)  # Shape: [seq_len, seq_len, batch_size]
            masked_matches_expanded = masked_matches.unsqueeze(-1).expand(-1, -1, -1, 1)
            # Step 5: Use masked_matches_expanded to index and sum the matched j positions
            # Multiply the mask with the j_ntoken_part to zero out non-matching elements
            j_nll_part = extended_one_hot[:, :, pos3+i-1:pos3+i]  # Shape: [seq_len, batch_size, 1]: this should be the nll at the matching position       
            expanded_j_nll_part = j_nll_part.unsqueeze(0).expand(seq_len, -1, -1, -1)  # Shape: [seq_len, seq_len, batch_size, 1]     
            masked_j_nll_part = expanded_j_nll_part * masked_matches_expanded  # Shape: [seq_len, seq_len, batch_size, 1]

            # Sum along the j dimension (dim=1) to get the sum of matched values
            sum_nll_values = masked_j_nll_part.sum(dim=1)  # Shape: [seq_len, batch_size, 1]

            # Step 6: Calculate the average by dividing the sum by the number of matches
            # Count the number of matches, ensuring to avoid division by zero
            match_counts = masked_matches.sum(dim=1).unsqueeze(-1) # Shape: [seq_len, batch_size, 1]
            average_nll_values = sum_nll_values / match_counts.clamp(min=1)

            # Step 7: Concatenate the average values as the 13th element in the original tensor along the v_dim dimension
            extended_one_hot[:, :,pos3+max_tree_depth+i:pos3+max_tree_depth+i+1] = average_nll_values # average nll for the matching positions            
        
        # Needs to process uni-gram separately because the matching will yield an empty tensor
        # The mask is simple this time        
        j_nll_part = extended_one_hot[:, :, pos3+max_tree_depth:pos3+max_tree_depth+1]  # Shape: [seq_len, batch_size, 1]: this should be the nll at the matching position       
        expanded_j_nll_part = j_nll_part.unsqueeze(0).expand(seq_len, -1, -1, -1)  # Shape: [seq_len, seq_len, batch_size, 1]     
        masked_j_nll_part = expanded_j_nll_part * mask_expanded[:,:,:,0:1]  # Shape: [seq_len, seq_len, batch_size, 1]        
        sum_nll_values = masked_j_nll_part.sum(dim=1)  # Shape: [seq_len, batch_size, self.ntoken]
        match_counts =  mask_expanded[:,:,:,0:1].sum(dim=1) # Shape: [seq_len, batch_size, 1]
        average_nll_values = sum_nll_values / match_counts.clamp(min=1)
        pos = pos3+max_tree_depth+max_tree_depth+1
        extended_one_hot[:, :, pos:pos+1] = average_nll_values              
        # This constructed layer 4: calculate the accumulative cross-entropy loss. The remaining network only needs a FF? 
        ##############################################################################################################################
        ##############################################################################################################################

        return extended_one_hot

class CustomTransformerMixedModel(nn.Module):

    def __init__(self, ntoken: int, max_prefix_len: int, d_model: int, nheads, d_hid: int,
                 nlayers: int, n_syn_layers: int, dropout: float = 0.5, need_weights = False, tfmode: str = 'normal'):
        super().__init__()
        self.n_layers = nlayers
        self.n_syn_layers = n_syn_layers
        self.need_weights = need_weights
        self.tfmode = tfmode

        assert n_syn_layers <= 4, "Only up to 4 synthetic layers are supported"

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)        
        self.transformer_encoder = CustomTransformerEncoder(d_model, nlayers, nheads, d_hid, dropout, need_weights, tfmode)
        # initialize both embedding
        self.custom_embedding = CustomEmbedding(ntoken, max_prefix_len, d_model, n_syn_layers, dropout, tfmode)  # custom embedding
        self.embedding = nn.Embedding(ntoken, d_model)                                                   # original embedding
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)                                         # Output mapping to token space

        self.init_weights()


    def init_weights(self) -> None:
        initrange = 0.1
        if self.n_syn_layers == 0:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.n_syn_layers > 0:
            src = self.custom_embedding(src) # Use custom embedding            
        else:
            src = self.embedding(src) * math.sqrt(self.d_model)         # Use standard embedding
        

        # If we construct the layers, then the positional encoding is disabled       
        if self.n_syn_layers < 1: 
            src = self.pos_encoder(src)        
        
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0)).to(device)
        output = self.transformer_encoder(src, src_mask)  
        output = self.linear(output)
        return output

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    # original definition without the need_weights flag
    # def __init__(self, *args, **kwargs):
    #     super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)
    #     self.attn_output_weights = None
    
    def __init__(self, d_model, nhead, need_weights, dim_feedforward=512, dropout=0.1):       
        super(CustomTransformerEncoderLayer, self).__init__( 
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout)
        self.need_weights = need_weights
        self.attn_output_weights = None
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu1 = nn.ReLU()
        #self.linear_middle = nn.Linear(dim_feedforward, dim_feedforward)
        #self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.attn_output_weights = self.self_attn(
            src, src, src, attn_mask = src_mask,
            key_padding_mask = src_key_padding_mask, need_weights = self.need_weights, 
            average_attn_weights = False # this will retrieve the attention weights for each head
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
   
    # def forward(self, src, src_mask=None, src_key_padding_mask=None):
    #     src2, self.attn_output_weights = self.self_attn(
    #         src, src, src, attn_mask = src_mask,
    #         key_padding_mask=src_key_padding_mask, need_weights=True, 
    #         average_attn_weights=False # this will retrieve the attention weights for each head
    #     )
    #     src = src + self.dropout1(src2)
    #     src = self.norm1(src)        

    #     src2 = self.linear1(src)
    #     src2 = self.relu1(src2)
    #     #######################################
    #     ### this middle layer is added to the default transformer to improve the FF capability
    #     src2 = self.linear_middle(src2)
    #     src2 = self.relu2(src2)
    #     # #######################################
    #     src2 = self.linear2(src2)
        
    #     src = src + self.dropout2(src2)
    #     src = self.norm2(src)
    #     return src

class CustomTransformerAttnOnlyEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead, need_weights, dim_feedforward=512, dropout=0.1):       
        super(CustomTransformerAttnOnlyEncoderLayer, self).__init__( 
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout)
        self.need_weights = need_weights
        self.attn_output_weights = None        

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.attn_output_weights = self.self_attn(
            src, src, src, attn_mask = src_mask,
            key_padding_mask = src_key_padding_mask, need_weights = self.need_weights, 
            average_attn_weights = False # this will retrieve the attention weights for each head
        )
        src = src + self.dropout1(src2)
        # remove normalization for attention only layer
        # src = self.norm1(src)        
        return src

class CustomTransformerAttnNormOnlyEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead, need_weights, dim_feedforward=512, dropout=0.1):       
        super(CustomTransformerAttnNormOnlyEncoderLayer, self).__init__( 
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout)
        self.need_weights = need_weights
        self.attn_output_weights = None        

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, self.attn_output_weights = self.self_attn(
            src, src, src, attn_mask = src_mask,
            key_padding_mask = src_key_padding_mask, need_weights = self.need_weights, 
            average_attn_weights = False # this will retrieve the attention weights for each head
        )
        src = src + self.dropout1(src2)        
        src = self.norm1(src)        
        return src
      

class CustomTransformerFFLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(CustomTransformerFFLayer, self).__init__(*args, **kwargs)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, heads_per_layer, dim_feedforward, dropout, need_weights = False, tfmode: str = 'normal'):
        super(CustomTransformerEncoder, self).__init__()
        self.need_weights = need_weights

        if tfmode == 'withoutFF':
            assert num_layers == 2, "Only 0 layer is supported for the withoutFF mode"
            self.layers = nn.ModuleList([
                # CustomTransformerEncoderLayer(
                #     d_model = d_model,
                #     nhead = heads_per_layer[0],
                #     need_weights = need_weights,
                #     dim_feedforward = dim_feedforward,
                #     dropout = dropout                    
                # ),
                CustomTransformerAttnOnlyEncoderLayer(
                    d_model = d_model,
                    nhead = heads_per_layer[0],
                    need_weights = need_weights,
                    dim_feedforward = dim_feedforward,
                    dropout = dropout                    
                ),
                CustomTransformerAttnOnlyEncoderLayer(
                    d_model = d_model,
                    nhead = heads_per_layer[1],
                    need_weights = need_weights,
                    dim_feedforward = dim_feedforward,
                    dropout = dropout                    
                )
            ])
            self.attention_weights = [None] * 2
        else:
            if num_layers != 0:
                if tfmode == 'normonly':
                    self.layers = nn.ModuleList([
                        #nn.TransformerEncoderLayer(
                        CustomTransformerAttnNormOnlyEncoderLayer(
                            d_model = d_model,
                            nhead = heads_per_layer[i],
                            need_weights = need_weights,
                            dim_feedforward = dim_feedforward,
                            dropout = dropout                    
                        )
                        for i in range(num_layers)
                    ])
                    self.attention_weights = [None] * num_layers
                else:
                    self.layers = nn.ModuleList([
                        #nn.TransformerEncoderLayer(
                        CustomTransformerEncoderLayer(
                            d_model = d_model,
                            nhead = heads_per_layer[i],
                            need_weights = need_weights,
                            dim_feedforward = dim_feedforward,
                            dropout = dropout                    
                        )
                        for i in range(num_layers)
                    ])
                    self.attention_weights = [None] * num_layers
            else:
                self.layers = nn.ModuleList([              
                    CustomTransformerFFLayer(
                        d_model = d_model,
                        nhead = 1,
                        dim_feedforward = dim_feedforward,
                        dropout = dropout
                    )
                ])
                self.attention_weights = [None] * 1
    def forward(self, src: Tensor, mask: Tensor = None):
        """
        Forward pass of the custom transformer encoder.

        Parameters:
        - src: The input sequence to the encoder [sequence length, batch size, d_model].
        
        Returns:
        - Output of the transformer encoder.
        """
        output = src

        for i, layer in enumerate(self.layers):
            output = layer(output, src_mask=mask)
            #if i == 0 and self.need_weights and hasattr(layer, 'attn_output_weights'):
                ###############################################################################################3
                # this is to collecting the statistics, is disabled during training and most of the evaluation                
                #for j in range(output.size(0)):
                #    print(output[j,0,:],flush=True)
                ###############################################################################################3
            if self.need_weights and hasattr(layer, 'attn_output_weights'):
                self.attention_weights[i] = layer.attn_output_weights

        return output

    def get_attention_weights(self):
        return self.attention_weights

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
