import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        """
        This class is a single head of self-attention. 
        It takes as input a tensor of size (batch, time-step, channels) 
        and returns a tensor of size (batch, time-step, head size) where 
        the head size is the size of the output of the attention mechanism.

        Args:
            head_size (_type_): _description_
            n_embd (_type_): _description_
            block_size (_type_): _description_
            dropout (_type_): _description_
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # bias is usually off
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # this is not a parameter and convention is to call it buffer
        # it's the matrix we are using to get our lower diagonal for attention
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # not used at this point
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        This function represents the forward pass of the self-attention mechanism.

        Args:
            x: this is the input of dimensions (B,T,C) where B is the batch size, 
            T is the time-step and C is the number of channels, which in this case
            is n_embed, the number of embeddings after generating an encoding
            that contains information about the characters/words in the input and
            the position of the characters/words in the input.

        Returns:
            tensor: (B, T, head_size) where B is the batch size, T is the time-step
            and head_size is the size of the output of the attention mechanism.
        """
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T), decoder block
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # not used at this point
        #wei = self.dropout(wei)
        
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
