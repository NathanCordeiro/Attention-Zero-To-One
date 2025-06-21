import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads): 
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # Ensure head_dim is an integer so do integer division

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        
        # Project inputs to Q, K, V
        qkv = self.qkv_proj(x).view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_length, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        return self.out_proj(attn_output)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # shape: (1, max_seq_length, embed_dim)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]  # Match (batch, seq_len, embed_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, 
                 max_seq_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm and output projection
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, seq_length):
        """Create a causal mask for decoder-style attention"""
        mask = torch.tril(torch.ones(seq_length, seq_length))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

    def forward(self, x, causal_mask=True):
        seq_length = x.size(1)
        
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask if needed (for autoregressive models)
        mask = None
        if causal_mask:
            mask = self.create_causal_mask(seq_length).to(x.device)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Final layer norm and output projection
        x = self.layer_norm(x)
        output = self.output_projection(x)
        
        return output


# Example usage and testing
if __name__ == "__main__":
    # Model hyperparameters
    vocab_size = 10000
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    ff_dim = 2048
    max_seq_length = 1024
    dropout = 0.1
    
    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    
    # Example input (batch_size=2, seq_length=10)
    input_ids = torch.randint(0, vocab_size, (2, 10))
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids)
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test individual components
    print("\n--- Testing Individual Components ---")
    
    # Test self-attention
    attention = SelfAttention(embed_dim=512, num_heads=8)
    test_input = torch.randn(2, 10, 512)
    attn_output = attention(test_input)
    print(f"Self-attention output shape: {attn_output.shape}")
    
    # Test positional encoding
    pos_enc = PositionalEncoding(embed_dim=512)
    pos_output = pos_enc(test_input)
    print(f"Positional encoding output shape: {pos_output.shape}")
    
    # Test feed-forward
    ff = FeedForward(embed_dim=512, ff_dim=2048)
    ff_output = ff(test_input)
    print(f"Feed-forward output shape: {ff_output.shape}")
    
    # Test transformer block
    transformer_block = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048)
    block_output = transformer_block(test_input)
    print(f"Transformer block output shape: {block_output.shape}")