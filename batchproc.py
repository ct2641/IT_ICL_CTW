
from torch import Tensor

# batching data for processing

def get_batch(source: Tensor, pos: int, bptt:int):
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    
    seq_len = min(bptt, len(source) - 1 - pos)
    data = source[pos:pos+seq_len]           
    # note the offset by 1 here, because the LLM needs to generate probabilistic model for the next word    
    target = source[pos+1:pos+1+seq_len].reshape(-1)    
    
    return data, target