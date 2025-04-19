import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from transformers import PreTrainedTokenizerFast, AddedToken
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders
import copy
import random
import math
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import re
import pandas as pd
import gc
from torch.nn.utils.rnn import pad_sequence
import typing
from torch.optim import Optimizer
import torch.utils.checkpoint as cp
import torch.distributed as dist
import torch.multiprocessing as mp
import pdfplumber


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return full_text

## set threads
torch.set_num_threads(64)
torch.set_num_interop_threads(64)

# Debug for CUDA

tokenizer = None
# Debug for CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
tokenizer = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

seq_len = 512  # Maximum sequence length for the model
pad_token_id = 0


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Uses a quintic iteration optimized for stability in low precision.
    """
    #print(f"Before NS: {G.shape}")

    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-4)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    del A, B, b, c
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized by Newton-Schulz

    A distributed-friendly optimizer that applies momentum-based updates and
    orthogonalization post-processing. Works on multi-GPU setups, but can also run
    in single-GPU mode by bypassing distributed operations.

    Arguments:
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        momentum: Momentum coefficient.
        nesterov: Use Nesterov-style momentum.
        ns_steps: Number of Newton-Schulz iterations.
        world_size: Number of GPUs used for distributed training.
        rank: Rank of the current process (set automatically in DDP).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5):
        # Detect whether distributed training is initialized
        self.ddp_enabled = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.ddp_enabled else 1
        self.rank = dist.get_rank() if self.ddp_enabled else 0

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)

        param_groups = []
        for size in {p.numel() for p in params}:
            # 🔹 Only create distributed buffers if DDP is enabled
            if self.ddp_enabled:
                b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="cuda")
                group = dict(params=[p for p in params if p.numel() == size],
                             update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
            else:
                group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = group["params"]

            if self.ddp_enabled:
                update_buffer: torch.Tensor = group["update_buffer"]
                update_buffer_views: list[torch.Tensor] = group["update_buffer_views"]
                handle = None
                params_world = None

            def update_prev():
                """Distributed update processing (only if DDP is enabled)."""
                if self.ddp_enabled:
                    handle.wait()
                    for p_world, g_world in zip(params_world, update_buffer_views):
                        p_world.mul_(1 - group["lr"] * group["weight_decay"])
                        p_world.add_(g_world.view_as(p_world),
                                    alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)

            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    #assert g is not None
                    if g is None:
                        continue  # skip this param

                    state = self.state[p]

                    # Initialize momentum buffer if not already present
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf: torch.Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                    # Handle convolutional filters
                    if g.ndim == 4:
                        g = g.view(len(g), -1)

                    # 🔹 DEBUG: Print before Newton-Schulz
                    #print(f"🔍 Before NS: {g.shape} (Original param shape: {p.shape})")

                    # 🔹 Fix potential reshape issue before NS
                    if g.ndim == 3:
                        g = g.view(g.shape[0], -1, g.shape[-1])  # Reshape 3D to 2D
                    elif g.ndim > 3:
                        g = g.view(g.shape[0], g.shape[1], -1)  # Handle extra dimensions

                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                    #print(f"✅ After NS: {g.shape}")

                else:
                    g = update_buffer_views[self.rank] if self.ddp_enabled else None

                # Handle distributed processing (skip if single GPU)
                if self.ddp_enabled:
                    if base_i > 0:
                        update_prev()
                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                    params_world = params[base_i: base_i + self.world_size]
                    torch.cuda.empty_cache()
                else:
                    # Apply updates directly if single-GPU
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                    torch.cuda.empty_cache()

            if self.ddp_enabled:
                update_prev()
                
def log_space_cross_entropy(logits, targets, epsilon=1e-30):
    """
    Compute cross-entropy loss in log space.
    """
    # Clamp logits to prevent log(0) or log(negative)
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logging.error("NaN or Inf detected in logits before log_softmax!")
        logging.error(f"Logits: {logits}")

    logits = torch.clamp(logits, min=epsilon)
    
    # Stabilize by subtracting the max value
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
    
    # Compute log softmax in log space
    log_probs = F.log_softmax(logits + epsilon, dim=-1)
    
    # Gather the log probabilities for the correct target labels
    loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return loss.mean()

def save_checkpoint(model, optimizer, epoch, phase, path):
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)
    return checkpoint['epoch'], checkpoint['phase']



def tokenize_chunk(chunk):
    # Tokenizer is now the global variable initialized in each process
    encoded = tokenizer(chunk, return_attention_mask=False, truncation=True, max_length=seq_len)
    return encoded['input_ids']

class RawPairDataset(torch.utils.data.Dataset):
    def __init__(self, query_target_pairs):
            self.pairs = query_target_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample = self.pairs[idx]
        if isinstance(sample, dict):
            return sample['query'], sample['target']
        return sample  # assume it's already a tuple

# Global tokenizer reference
global_tokenizer = None
def init_collate_globals(tokenizer):
    global global_tokenizer
    global_tokenizer = tokenizer

def worker_init_fn(worker_id):
    global global_tokenizer
    tokenizer_path = r"C:\Users\abias\.cursor-tutor\vccdoe\mhlamodel\mhla\models\slidingcontext1256"
    global_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

def collate_fn_global(batch):
    global global_tokenizer

    BOS = global_tokenizer.bos_token or "<BOS>"
    EOS = global_tokenizer.eos_token or "<EOS>"
    PAD_ID = global_tokenizer.pad_token_id or 0

    def encode(text):
        return global_tokenizer.encode(BOS + " " + text + " " + EOS, add_special_tokens=False)

    if isinstance(batch[0], tuple):
        full_seqs = []
        for query, target in batch:
            query_ids = global_tokenizer.encode(query, add_special_tokens=False)
            target_ids = global_tokenizer.encode(target, add_special_tokens=False)

            full_seq = query_ids + target_ids 
            full_seqs.append(torch.tensor(full_seq, dtype=torch.long))

        padded_batch = pad_sequence(full_seqs, batch_first=True, padding_value=PAD_ID)  # [batch, max_len]
        return padded_batch

    elif isinstance(batch[0], str):
        token_ids = [torch.tensor(encode(t)) for t in batch]
        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=PAD_ID)
        return token_ids


class TokenizerWrapper:
    def __init__(self, tokenizer, seq_len=seq_len, add_bos=True, add_eos=True, pad_to_max=True, shift_decoder=False, device=device):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.pad_to_max = pad_to_max
        self.shift_decoder = shift_decoder
        self.device = device

        self.bos_token = tokenizer.bos_token or "<BOS>"
        self.eos_token = tokenizer.eos_token or "<EOS>"
        self.pad_token_id = tokenizer.pad_token_id or 0

    def format(self, text):
        if isinstance(text, list):
            return [self.format(t) for t in text]
        return f"{self.bos_token} {text} {self.eos_token}" if self.add_bos and self.add_eos else text

    def encode(self, text_batch, truncate=True):
        if isinstance(text_batch[0], str):
            text_batch = self.format(text_batch)

        encoded = [self.tokenizer.encode(t, add_special_tokens=False) for t in text_batch]
        result = []
        for tokens in encoded:
            if truncate and len(tokens) > self.seq_len:
                tokens = tokens[:self.seq_len - 1] + [self.tokenizer.eos_token_id]
            result.append(tokens)
        return result if not self.pad_to_max else torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, device=self.device) for seq in result],
            batch_first=True,
            padding_value=self.pad_token_id
        )

    def encode_shifted_pair(self, text_batch):
        """Returns (decoder_input_ids, labels), both padded"""
        full = self.encode(text_batch)  # [B, T]
        decoder_input = full[:, :-1]
        labels = full[:, 1:]
        return decoder_input, labels


# In your collate_fn, specify device when creating new tensors:
device_cpu = 'cpu'

# Generate src mask function
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_combined_mask(batch_input_ids, pad_token_id):
    """
    Create a combined attention mask that incorporates both the causal (subsequent) mask
    and the padding mask. This function ensures that each row has at least one valid token.
    """
    batch_size, seq_length = batch_input_ids.size()
    device = batch_input_ids.device
    
    # Generate causal (subsequent) mask: shape (seq_len, seq_len)
    causal_mask = generate_square_subsequent_mask(seq_len).to(device)
    logging.debug(f"Shape of causal_mask before expand: {causal_mask.shape}")

    # Expand to batch dimension: (batch_size, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    logging.debug(f"Shape of causal_mask after expansion: {causal_mask.shape}")
    # Create padding mask: valid tokens are True, padded tokens are False.
    # Shape: (batch_size, seq_len)
    padding_mask = (batch_input_ids != pad_token_id)
    # Expand padding mask to match the shape (batch_size, seq_len, seq_len)
    # Here we broadcast along one dimension so that we mask out positions in each row.
    padding_mask_expanded = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
    logging.debug(f"Shape of padding_mask after expansion: {padding_mask_expanded.shape}")

    # Combine masks: where padding_mask is False, set to -inf.
    # This keeps the causal structure while ensuring that padded positions are fully masked.
    combined_mask = causal_mask.masked_fill(~padding_mask_expanded, float('-inf'))
    logging.debug(f"Shape of combined_mask after fill: {combined_mask.shape}")

    # Check each row: if an entire row is -inf, force the first token (or a designated position) to be valid.
    for i in range(batch_size):
        for j in range(seq_len):
            if torch.all(combined_mask[i, j] == float('-inf')):
                combined_mask[i, j, 0] = 0.0  # Force at least one valid position
    
    return combined_mask

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent):
        """
        Multi-Head Latent Attention (MHLA)
        - d_model: Input feature dimension
        - num_heads: Number of attention heads
        - d_latent: Compressed latent space dimension
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent

        # Standard attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Latent compression & reconstruction
        self.W_down_kv = nn.Linear(d_model, d_latent, bias=False)  # Compress keys/values
        self.W_up_k = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct keys
        self.W_up_v = nn.Linear(d_latent, d_model, bias=False)  # Reconstruct values

    def forward(self, x, memory=None):
        """
        Forward pass with optional memory (for hierarchical tokenization)
        - x: Input tensor (batch, seq_len, d_model)
        - memory: Cached latent state (batch, d_latent) [optional]
        """
        batch_size, seq_len, _ = x.shape

        # Compute queries, keys, values
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Latent compression for keys and values
        latent_kv = self.W_down_kv(k + v)  # Merge and compress
        if memory is not None:
            latent_kv = (latent_kv + memory) / 2
            latent_kv = torch.nan_to_num(latent_kv)

        # Reconstruct full-size keys and values
        k_reconstructed = self.W_up_k(latent_kv)
        v_reconstructed = self.W_up_v(latent_kv)

        # Multi-head split
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k_reconstructed = k_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v_reconstructed = v_reconstructed.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn_weights = torch.matmul(q, k_reconstructed.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=10.0, neginf=-10.0)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reconstructed)


        # Merge attention heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final projection
        output = self.W_o(attn_output)

        return output, latent_kv  # Return output and memory for next layer



#Saved for reference or later implementation as an option
class MiniTransformerNode(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length, tokenizer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.vocab_size=vocab_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.seq_length = max_seq_length
        self.num_heads = num_heads
        self.device = device
        self.pos_encoder = RotaryPositionalEmbedding(embed_size)
        #self.pos_encoder = DynamicPositionalEncoding(embedding_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoderLayer(d_model=embed_size, dim_feedforward=hidden_size, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=num_layers)
        self.decoder = nn.TransformerDecoderLayer(d_model=embed_size, dim_feedforward=hidden_size, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.decoder_layers = nn.TransformerDecoder(decoder_layer=self.decoder, num_layers=num_layers)
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, seq_len=max_seq_length, shift_decoder=False, device=device)
        self.tokenizer = tokenizer
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.cross_node_attention = MultiHeadLatentAttention(embed_size, num_heads, d_latent=embed_size // 2)

    def generate_mask(self, src, tgt):
        # Padding mask: (batch_size, seq_len) with True for padding tokens
        src_pad_mask = (src == 0)  # Shape: [batch, src_len]
        tgt_pad_mask = (tgt == 0)  # Shape: [batch, tgt_len]

        # Causal mask for decoder (no peeking into the future)
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(self.device)  # Shape: [tgt_len, tgt_len]

        return src_pad_mask, tgt_pad_mask, causal_mask

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    def forward(self, src, tgt_ids, prev_node_output=None, src_mask=None, is_final_node=False, latent_memory=None, mode="eval"):
        if isinstance(src[0], str):
            src = self.tokenizer_wrapper.encode(src)
        if tgt_ids is not None and isinstance(tgt_ids[0], str):
            tgt_ids= self.tokenizer_wrapper.encode(tgt_ids)
        elif tgt_ids is not None and mode == "eval":
            tgt_ids = tgt_ids
            #tgt_ids = tgt_ids[:, 1:]
            #print(f"\n🚀 FORWARD: src shape {src.shape}, tgt shape {tgt_ids.shape}")
        elif tgt_ids is not None and tgt_ids.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")
        if src.dim() == 2:  
            # Check before using tgt_ids in the model
            if src.max() >= self.tokenizer.vocab_size:
                print("⚠️ Warning: Token ID exceeds vocab range.")
                src = src.clamp(0, self.tokenizer.vocab_size - 1)
            #print(f"src shape main call: {src.shape} ")
            src_emb = self.embedding(src)  # Shape: (batch_size, seq_length, embed_size)
            src_mask, tgt_pad_mask, causal_mask = self.generate_mask(src, tgt_ids if tgt_ids is not None else src)
        else:  
            #print(f"backup src call src shape: {src.shape} ")
            src_emb = src # If already embeddings, use them
            _, tgt_pad_mask, causal_mask = self.generate_mask(src, tgt_ids if tgt_ids is not None else src)

        # Ensure x is within the correct token index range
        src = src.to(device)
        # Check before using tgt_ids in the model

        src_emb = self.pos_encoder(src_emb)
        def layer_fn(*inputs):
            return self.encoder_layers(
                inputs[0], 
                src_key_padding_mask=inputs[1]
            )
        memory = layer_fn(src_emb, src_mask)

        # Cross-node attention (global attention) - apply only if there is a previous node
        if prev_node_output is not None:
            tgt_emb, latent_memory= self.cross_node_attention(prev_node_output, memory=latent_memory)

        else:
            if tgt_ids is None:
                tgt_ids = src[:, :1]  # dummy start
            if tgt_ids.max() >= self.tokenizer.vocab_size:
                print("⚠️ Warning: Token ID exceeds vocab range.")
                tgt_ids = tgt_ids.clamp(0, self.tokenizer.vocab_size - 1)

            tgt_emb = self.embedding(tgt_ids)
            tgt_emb = self.pos_encoder(tgt_emb)

            #print(f"💡 Embeddings: src {src_emb.shape}, tgt {tgt_emb.shape}")


        def decoder_layer_fn(*inputs):
            return self.decoder_layers(
                inputs[0], memory,
                tgt_mask=inputs[1],
                tgt_key_padding_mask=inputs[2],
                memory_key_padding_mask=inputs[3]
            )
        output = decoder_layer_fn(tgt_emb, causal_mask, tgt_pad_mask, src_mask)
        # Forward through transformer encoder (self-attention)


        # Final token prediction layer in the final node
        if is_final_node:
            output = self.fc_out(output)


        return output, src_mask, latent_memory

#Standard Node network
class CascadeTransformer(nn.Module):
    def __init__(self, num_nodes, vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length, tokenizer):
        super().__init__()
        self.nodes = nn.ModuleList([
            MiniTransformerNode(vocab_size, embed_size, num_heads, num_layers, hidden_size, max_seq_length, tokenizer) 
            for _ in range(num_nodes)
        ])
        self.num_heads=num_heads
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, seq_len=max_seq_length, shift_decoder=False, device=device)
        self.tokenizer = tokenizer
    
    def forward(self, x, tgt,  mask=None, mode="eval"):
        
        prev_node_output = None
        latent_memory = None
        for i, node in enumerate(self.nodes):
            is_final_node = (i == len(self.nodes) - 1)
            x, mask, latent_memory= node(x, tgt, prev_node_output=prev_node_output, src_mask=mask, is_final_node=is_final_node, latent_memory=latent_memory, mode=mode)
            prev_node_output = x

        return x

def pad_mask(mask, target_size):
    """
    Pad the mask to match the required size.
    Args:
        mask (torch.Tensor): Original mask of shape (seq_len-1, seq_len-1)
        target_size (int): The target size to pad to (e.g., seq_len)
    Returns:
        torch.Tensor: Padded mask of shape (target_size, target_size)
    """
    pad_size = target_size - mask.size(0)
    if pad_size > 0:
        # Pad with -inf on the last row and column
        padding = torch.full((mask.size(0), pad_size), float('-inf'), device=mask.device)
        mask = torch.cat([mask, padding], dim=1)
        padding = torch.full((pad_size, target_size), float('-inf'), device=mask.device)
        mask = torch.cat([mask, padding], dim=0)
    return mask

def causal_mask(seq_len):
    """
    Creates a mask to prevent attending to future tokens.
    Args:
        seq_len (int): Length of the sequence
    Returns:
        mask (torch.Tensor): Shape [seq_len, seq_len], lower triangular matrix
    """
    return torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0)  # Add batch dimension

def padding_mask(input_ids, pad_token_id=pad_token_id):
    """
    Creates a mask for padded tokens in a batch.
    Args:
        input_ids (torch.Tensor): Shape [batch_size, seq_len]
        pad_token_id (int): Token ID representing padding (default 0)
    Returns:
        mask (torch.Tensor): Shape [batch_size, seq_len, seq_len]
    """
    mask = (input_ids != pad_token_id).unsqueeze(1).expand(-1, input_ids.size(1), -1)
    return mask


def create_memory_mask(memory, pad_token_id=pad_token_id):
    """
    Creates a memory mask for encoder-decoder attention.
    Masks padding tokens in the encoder output.
    Args:
        memory (torch.Tensor): Shape [batch_size, seq_len, d_model]
        pad_token_id (int): ID representing padding (usually 0)
    Returns:
        mask (torch.Tensor): Shape [batch_size, 1, seq_len]
    """
    return (memory != pad_token_id)  # Shape: [batch_size, 1, seq_len]

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model


    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, d_model]

        return self.dropout(x + pe)

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(x, sinusoidal_emb):
    return (x * sinusoidal_emb.cos()) + (rotate_half(x) * sinusoidal_emb.sin())

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)[None, :, :]  # [1, seq_len, dim]
        return apply_rotary(x, emb)

    
class Transformer_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, seq_length, tokenizer, device=device):
        super().__init__()
        self.embed_size = embedding_dim
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.device = device
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = RotaryPositionalEmbedding(embedding_dim)
        #self.pos_encoder = DynamicPositionalEncoding(embedding_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=embedding_dim, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer=self.encoder, num_layers=num_layers)
        self.decoder = nn.TransformerDecoderLayer(d_model=embedding_dim, dim_feedforward=embedding_dim, nhead=num_heads, activation="gelu", batch_first=True, device=device)
        self.decoder_layers = nn.TransformerDecoder(decoder_layer=self.decoder, num_layers=num_layers)
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, seq_len=10000, shift_decoder=False, device=device)
        self.tokenizer = tokenizer
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def generate_mask(self, src, tgt):
        # Padding mask: (batch_size, seq_len) with True for padding tokens
        src_pad_mask = (src == 0)  # Shape: [batch, src_len]
        tgt_pad_mask = (tgt == 0)  # Shape: [batch, tgt_len]

        # Causal mask for decoder (no peeking into the future)
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(self.device)  # Shape: [tgt_len, tgt_len]

        return src_pad_mask, tgt_pad_mask, causal_mask

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


    def forward(self, src, tgt_ids=None, mode='eval'):

        if isinstance(src[0], str):
            src = self.tokenizer_wrapper.encode(src)
        if tgt_ids is not None and isinstance(tgt_ids[0], str):
            tgt_ids= self.tokenizer_wrapper.encode(tgt_ids)
        elif tgt_ids is not None and mode == 'train':
            tgt_ids = tgt_ids
            #tgt_ids = tgt_ids[:, 1:]
        #print(f"\n🚀 FORWARD: src shape {src.shape}, tgt shape {tgt_ids.shape}")
        elif tgt_ids is not None and tgt_ids.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")

        src_pad_mask, tgt_pad_mask, causal_mask = self.generate_mask(src, tgt_ids if tgt_ids is not None else src)
        #print(f"📏 src_pad_mask: {src_pad_mask.shape}")
        #print(f"📏 tgt_pad_mask: {tgt_pad_mask.shape}")
        #print(f"📏 causal_mask: {causal_mask.shape}")

        src_emb = self.token_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        def layer_fn(*inputs):
            return self.encoder_layers(
                inputs[0], 
                src_key_padding_mask=inputs[1]
            )
        memory = cp.checkpoint(layer_fn, src_emb, src_pad_mask)
            
        if tgt_ids is None:
            tgt_ids = src[:, :1]  # dummy start

        tgt_emb = self.token_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)
        #print(f"💡 Embeddings: src {src_emb.shape}, tgt {tgt_emb.shape}")

        def decoder_layer_fn(*inputs):
            return self.decoder_layers(
                inputs[0], memory,
                tgt_mask=inputs[1],
                tgt_key_padding_mask=inputs[2],
                memory_key_padding_mask=inputs[3]
            )
        output = cp.checkpoint(decoder_layer_fn, tgt_emb, causal_mask, tgt_pad_mask, src_pad_mask)

        return self.fc_out(output)

    
def sample_gumbel(shape, eps=1e-20, device='cpu'):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [*, num_classes] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but still use softmax gradients
    Returns:
        [*, num_classes] sample from the Gumbel-Softmax distribution.
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        # Straight-through trick: make hard one-hot output, but keep soft gradients
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        # Set gradients of y_hard equal to those of y
        y = (y_hard - y).detach() + y
    logging.debug(f"Gumbel shape: {y.shape}") 

    return y

def greedy_sample(logits):
    """ Converts raw model outputs into discrete tokens using greedy sampling. """
    probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities
    return torch.argmax(probs, dim=-1)  # Select the most probable token



class GeneticAlgorithm:
    def __init__(self, model, mutation_rate, population_size=10):
        self.model = model
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [self._randomize_weights() for _ in range(population_size)]

    def _randomize_weights(self):
        new_model = copy.deepcopy(self.model)
        for param in new_model.parameters():
            param.data += torch.randn_like(param) * self.mutation_rate  # Mutate weights
        return new_model

    def select_best(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        best_model = None
        best_loss = float('inf')
        n=0
        loss = 0
        if architecture == "Cascade Transformer":

            output = self.model(inputs, decoder_input)

        else:
            output = self.model(inputs, decoder_input)          
                
        output = output.reshape(-1, output.shape[-1])
        logging.debug(f"output reshaped Shape: {output.shape}")
        loss = loss_fn(output, target_labels)
        best_loss = loss
        print(f"Original model iteration {n}, Loss: {loss.item()}")
        best_model = self.model
        for model in self.population:
            loss = 0
            if architecture == "Cascade Transformer":

                output = model(inputs, decoder_input)

            else:
                output = model(inputs, decoder_input)          
                
            output = output.reshape(-1, output.shape[-1])
            logging.debug(f"output reshaped Shape: {output.shape}")
            loss = loss_fn(output, target_labels)
            if loss < best_loss:
                    best_loss = loss
                    n=n+1
                    print(f"Best model iteration {n}, Loss: {loss.item()}")
                    best_model = model
            
            else:
                loss = 0

                if architecture == "Cascade Transformer":

                    output = model(inputs, decoder_input)

                else:
                    output = model(inputs, decoder_input)
                # Flatten logits and targets:
                output = output.reshape(-1, output.shape[-1])
                logging.debug(f"output reshaped Shape: {output.shape}")
                loss = loss_fn(output, target_labels)
                n=n+1
                print(f"Iteration {n}, Loss: {loss}")
                if loss < best_loss:
                        best_loss = loss
                        n=n+1
                        print(f"Best model iteration {n}, Loss: {loss.item()}")
                        best_model = model
        return best_model

    def evolve(self, loss_fn, inputs, target_labels, decoder_input, architecture):
        self.model = self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)
        self.population = [copy.deepcopy(self.model) for _ in range(self.population_size)]
        for model in self.population:
            for param in model.parameters():
                param.data += torch.randn_like(param) * self.mutation_rate  # Apply mutation
        # Return the best model from the new population.
        return self.select_best(loss_fn, inputs, target_labels, decoder_input, architecture)


def generate_text(model, input_ids, max_length, tokenizer, temperature=1.2, top_k=40, repetition_penalty=1.2):
    model.eval()
    generated = input_ids

    for _ in range(max_length):
        # Forward pass through the model
        outputs = model(generated, generated)
        next_token_logits = outputs[:, -1, :]  # Get logits for the last token
        
        # Apply repetition penalty while excluding special tokens
        for token in set(generated[0].tolist()):
            if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                next_token_logits[0, token] /= repetition_penalty
        
        # Temperature scaling
        next_token_logits /= temperature
        
        # Top-k Sampling
        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        next_token = top_k_indices.gather(dim=-1, index=torch.multinomial(top_k_probs, num_samples=1))
        
        # Append the newly generated token to the sequence
        generated = torch.cat((generated, next_token), dim=1)

        # Stop if end-of-sequence token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0].tolist())

def prepare_decoder_input_and_target(target):
    """
    Prepares inputs and targets for teacher forcing when <BOS> is auto-generated by the tokenizer.
    - target: Tensor of shape (batch_size, seq_len)
    Returns:
    - decoder_input: Shifted target, including <BOS>
    - target_output: Original target
    """
    # Shift target to the right to form the decoder input
    decoder_input = torch.zeros_like(target)
    decoder_input[:, 1:] = target[:, :-1]  # Shift right
    decoder_input[:, 0] = target[:, 0]     # Copy the <BOS> from the target

    # The output is the target sequence itself (including <EOS>)
    target_output = target
    
    return decoder_input, target_output

def build_custom_validation_batch(tokenizer, seq_len=seq_len, device=device, batch_size=1):
    query_strings = [
        "1. What is 17 + 35?",
        "2. Solve for x: 2x + 5 = 13",
        "3. What is the derivative of x^2?",
        "4. What is the integral of x dx?",
        "5. What is the plural of 'analysis'?",
        "6. Is this sentence correct? 'He go to school every day.'",
        "7. What is the first law of Robotics?",
        "8. What is the secpnd law of robotics?",
        "9. What is the third law of robotics?,",
        "10. What is the zeroth law of robotics?",
        "11. What does this Python function return? def square(x): return x * x",
        "12. Write a function in Python that checks if a number is prime.",
        "13. What is the derivative of a function x^3 according to calculus?",
        "14. Describe the integral of a function x^3 according to calculus, please."
    ]

    target_strings = [
        "1. 52",
        "2. x = 4",
        "3. 2x",
        "4. (1/2)x^2 + C",
        "5. analyses",
        "6. No, it should be: 'He goes to school every day.'",
        "7. 1. A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
        "8. 2. A robot must obey orders given by humans except where such orders would conflict with the First Law.",
        "9. 3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
        "10. 0. A robot may not harm humanity, or, by inaction, allow humanity to come to harm.",
        "11. It returns the square of x.",
        "12. def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "13. The derivative of x^3 by the power law for derivatives would be 3x^2.",
        "14. According to the integral power law the integral of x^3 would be (x^2)/2."
    ]

    input_ids, target_ids = [], []
    for query, target in zip(query_strings, target_strings):
        q_ids = tokenizer.encode(query, max_length=seq_len, truncation=True, padding='max_length')
        a_ids = tokenizer.encode(target, max_length=seq_len, truncation=True, padding='max_length')

        input_ids.append(q_ids)
        target_ids.append(a_ids)

    input_tensor = torch.tensor(input_ids[:batch_size], device=device)
    target_tensor = torch.tensor(target_ids[:batch_size], device=device)
    return input_tensor, target_tensor

class ReasoningModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reasoning Model GUI")

        # Transformer Parameters
        self.layers = []
        # Model Configuration Variables
        self.model_name = tk.StringVar(value="Reasoning Model")
        self.num_parameters = tk.IntVar(value=1024)
        self.num_heads = tk.IntVar(value=4)
        self.vocab_size = tk.IntVar(value=10000)
        self.hidden_size = tk.IntVar(value=8)
        self.num_layers = tk.IntVar(value=2)
        self.num_nodes = tk.IntVar(value=36)


        self.pad_token_id = 0  # Default value, adjust based on your tokenizer setup

        # Device Selection Variable
        self.device_option = tk.StringVar(value="GPU" if torch.cuda.is_available() else "CPU")

        # Dynamically calculate parameters based on other inputs
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # Set initial calculated value
        self.update_num_parameters()

        # Training Parameters
        self.dataset_path = ""
        self.vocab_path = ""
        self.tokenizer_path = ""
        self.batch_size = tk.IntVar(value=1)
        self.learning_rate = tk.DoubleVar(value=0.0005)
        self.epochs = tk.IntVar(value=10)

        # Training Variables
        self.loss_history = []
        self.accuracy_history = []
        self.current_epoch = 0
        self.stop_training = threading.Event()

        # Model and Data Variables
        self.model = None
        self.tokenizer = None
        self.dataset_path = None
        self.vocab_path = None
        self.tokenizer_path = None
        self.model_path = None
        self.train_data = None  # To store the dataset
        self.use_genetic_algo = "GHR Optim"  # default to optim
        self.validation_loader = None
        
        # Device (CPU or GPU) - Initially set based on device_option
        self.device = torch.device(self.map_device(self.device_option.get()))

        # Select log file path
        self.select_log_file()

        # Setup logging
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        logging.info(f"Using device: {self.device}")

        self.create_widgets()

    def map_device(self, selected_device):
        device_mapping = {
            "CPU": "cpu",
            "GPU": "cuda"
        }
        return device_mapping.get(selected_device, "cpu")

    def create_widgets(self):
        # Transformer Construction Frame
        transformer_frame = ttk.LabelFrame(self.root, text="Transformer Construction", padding=(10, 10))
        transformer_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(transformer_frame, text="Number of Parameters:").grid(row=0, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_parameters, state="readonly").grid(row=0, column=1)

        ttk.Label(transformer_frame, text="Number of Heads:").grid(row=1, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_heads).grid(row=1, column=1)
        
        ttk.Label(transformer_frame, text="Vocabulary Size:").grid(row=2, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.vocab_size).grid(row=2, column=1)

        ttk.Label(transformer_frame, text="Hidden Size:").grid(row=3, column=0, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.hidden_size).grid(row=3, column=1)

        ttk.Label(transformer_frame, text="Number of Layers:").grid(row=2, column=4, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_layers).grid(row=2, column=5)
        
        ttk.Label(transformer_frame, text="Number of Nodes:").grid(row=3, column=3, sticky="w")
        ttk.Entry(transformer_frame, textvariable=self.num_nodes).grid(row=3, column=4)
        
        # Device Selection
        ttk.Label(transformer_frame, text="Select Device:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        device_options = ["CPU"]
        if torch.cuda.is_available():
            device_options.append("GPU")
        device_combo = ttk.Combobox(transformer_frame, textvariable=self.device_option, values=device_options, state="readonly")
        device_combo.grid(row=4, column=1, sticky="w", pady=(10, 0))
        device_combo.bind("<<ComboboxSelected>>", self.on_device_change)

        # Attach parameter calculation to variable updates
        self.vocab_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.hidden_size.trace_add("write", lambda *args: self.update_num_parameters())
        self.num_layers.trace_add("write", lambda *args: self.update_num_parameters())

        # For resuming training
        ttk.Button(transformer_frame, text="Select Model File", command=self.select_model_file).grid(row=3, column=2, pady=5)

        # Architecture selection
        self.architecture = tk.StringVar(value="Cascade Transformer")
        ttk.Label(transformer_frame, text="Select Architecture:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.architecture, values=["Reasoning Model", "Cascade Transformer"], state="readonly").grid(row=0, column=3)

        ttk.Button(transformer_frame, text="Save Transformer and Model", command=self.save_model).grid(row=1, column=3, pady=5)
        ttk.Button(transformer_frame, text="Initialize Model", command=self.load_model).grid(row=2, column=3, pady=5)
        self.genetic_algo_var = tk.StringVar(value="GHR Optim")
        ttk.Label(transformer_frame, text="Algo:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.genetic_algo_var, values=["GHR Optim", "Genetic Algorithm"], state="readonly").grid(row=0, column=4)

        # Data Selection Frame
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=(10, 10))
        data_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(data_frame, text="Select Tokenizer", command=self.select_tokenizer).pack(pady=5)
        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)

        # Training Configuration Frame
        train_frame = ttk.LabelFrame(self.root, text="Training Configuration", padding=(10, 10))
        train_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.batch_size).grid(row=0, column=1)

        ttk.Label(train_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.learning_rate).grid(row=1, column=1)

        ttk.Label(train_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
        ttk.Entry(train_frame, textvariable=self.epochs).grid(row=2, column=1)

        ttk.Button(train_frame, text="Start Training", command=self.start_training).grid(row=3, column=0, pady=5)
        ttk.Button(train_frame, text="Save Model", command=self.save_model).grid(row=3, column=1, pady=5)
        ttk.Button(train_frame, text="Stop Training", command=self.stop_training_command).grid(row=4, column=0, pady=5)
        self.training_mode = tk.StringVar(value="response")  # Default
        training_modes = ["imitation", "completion", "response"]
        ttk.Combobox(data_frame, textvariable=self.training_mode, values=training_modes, state="readonly").pack(pady=5)
        #ttk.Button(train_frame, text="Run Validation", command=self.run_validation_button).grid(row=5, column=0, pady=5)
        ttk.Button(train_frame, text="Test Inference", command=self.test_inference).grid(row=4, column=1, pady=5)

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        self.status_label = ttk.Label(self.root, text="Status: Ready")
        self.status_label.pack(pady=5)

    def select_log_file(self):
        self.log_file_path = filedialog.asksaveasfilename(
            title="Select Log File Location",
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")]
        )
        if self.log_file_path:
            print(f"Log file will be saved to: {self.log_file_path}")
        else:
            self.log_file_path = 'training_debug.log'  # Default log file
            print(f"No log file selected. Using default: {self.log_file_path}")

    def calculate_parameters(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        embedding_params = vocab_size * embedding_dim * 4  # Quaternion embeddings (4x normal embedding size)
        transformer_params = num_layers * (
            (embedding_dim * hidden_dim * 4) +  # Attention layers
            (hidden_dim * hidden_dim * 4) +  # Feed-forward layers
            (hidden_dim * 4 * embedding_dim * 4)  # Output layers
        )
        output_projection_params = embedding_dim * 4 * vocab_size  # Final projection
        return embedding_params + transformer_params + output_projection_params

    def test_inference(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return

        # Set the model to evaluation mode
        self.model.eval()
        
        # Prompt the user for input text
        prompt = simpledialog.askstring("Test Inference", "Enter input text:")
        if prompt:
            try:
                if self.architecture.get() == "Cascade Transformer":
                    generated = []
                    tokenizer = self.tokenizer
                    max_generated=120
                    repetition_penalty=1.2
                    top_p=0.9
                    max_tokens = 50
                    model = self.model
                    model.eval()
                    generated_tokens = []

                    with torch.no_grad():
                        # Tokenize prompt → fixed encoder input
                        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

                        # Encoder is static throughout generation
                        encoder_input_ids = input_ids

                        # Setup initial decoder input
                        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
                        tgt_ids = torch.tensor([[bos_token_id]], device=device)

                        for _ in range(max_generated):
                            # Forward pass through model
                            batch_size, seq_lengths = encoder_input_ids.size()
                            encoder_input_len = encoder_input_ids.size(1)
                            tgt_ids_len = tgt_ids.size(1)
                            if tgt_ids_len < encoder_input_len:
                                pad_len = encoder_input_len - tgt_ids_len
                                pad_token_id = tokenizer.pad_token_id or 0
                                padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
                                tgt_ids = torch.cat([tgt_ids, padding], dim=1)
                            # Pad encoder input to max model length
                            elif encoder_input_len < tgt_ids_len:
                                pad_len = tgt_ids_len - encoder_input_len
                                pad_token_id = tokenizer.pad_token_id or 0
                                padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
                                encoder_input_ids = torch.cat([encoder_input_ids, padding], dim=1)

                            outputs= model(encoder_input_ids, tgt_ids)
                            logits = outputs[:, -1, :]  # (batch, vocab)

                            # Repetition penalty
                            for token in set(tgt_ids[0].tolist()):
                                if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]:
                                    logits[0, token] /= repetition_penalty

                            # Top-p sampling
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            filtered_logits = logits.clone()
                            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')

                            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                            # Stop at EOS
                            if next_token_id.item() == tokenizer.eos_token_id:
                                break

                            # Append and continue
                            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)
                            generated.append(self.tokenizer.decode(next_token_id.item()))

                            # Pad if needed to align with model
                            if tgt_ids.size(1) > seq_len:
                                tgt_ids = tgt_ids[:, -seq_len:]
                            
                else:
                    max_generated = 50
                    generated_tokens = []
                    generated = []
                    repetition_penalty = 1.2  # Adjust for stronger penalty
                    top_p = 0.9  # Cumulative probability threshold

                    self.model.eval()
                    with torch.no_grad():
                        input_ids = self.model.tokenizer_wrapper.encode([prompt], truncate=True)
                        src_tokens = input_ids[0]
                        if isinstance(src_tokens, torch.Tensor):
                            src_tokens = src_tokens.tolist()
                        src_tokens = src_tokens[:self.model.tokenizer_wrapper.seq_len]

                        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)

                        bos_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id or 0
                        eos_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id or 1

                        decoder_tokens = torch.tensor([[bos_id]], dtype=torch.long, device=device)
                        generated_tokens = [bos_id]

                        for step in range(max_generated):
                            logits = self.model(src_tensor, decoder_tokens)
                            next_token_logits = logits[:, -1, :]
                            # Top-p sampling
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            filtered_logits = next_token_logits.clone()
                            filtered_logits[0, sorted_indices[0][sorted_indices_to_remove[0]]] = float('-inf')
                            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                            generated_tokens.append(next_token_id)

                            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                            decoder_tokens = torch.cat([decoder_tokens, next_token_tensor], dim=1)


                            print(f"[{step}] Input: {self.tokenizer.decode(decoder_tokens[0])}, Next: {self.tokenizer.decode(next_token_id[0])}")
                            generated.append(self.tokenizer.decode(next_token_id[0].tolist()))

                            if next_token_id.item() == eos_id:
                                break
                            
                messagebox.showinfo("Inference Output", generated)
                logging.info(f"Inference Output: {generated}")

            except Exception as e:
                messagebox.showerror("Error", f"Inference failed: {str(e)}")
                logging.error(f"Inference failed: {str(e)}")

        # Optionally, return to train mode if further training is planned
        self.model.train()

    def update_num_parameters(self):
        vocab_size = self.vocab_size.get()
        embed_size = self.hidden_size.get()
        num_layers = self.num_layers.get()
        hidden_size = self.hidden_size.get()

        total_params = self.calculate_parameters(vocab_size, embed_size, num_layers, hidden_size)
        self.num_parameters.set(total_params)

    def on_device_change(self, event):
        selected_device = self.device_option.get()
        if selected_device == "GPU" and not torch.cuda.is_available():
            messagebox.showerror("Error", "GPU selected but CUDA is not available on this system.")
            self.device_option.set("CPU")
            selected_device = "CPU"
        device_str = self.map_device(selected_device)
        self.device = torch.device(device_str)
        logging.info(f"Device changed to: {self.device}")
        messagebox.showinfo("Device Selection", f"Computation device set to: {selected_device}")

    def select_model_file(self):
        try:
            if not self.tokenizer:
                messagebox.showerror(
                    title="Tokenizer Required",
                    message="Please select a tokenizer before loading a model file."
                )
                return

            self.model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("Model Files", "*.pth;*.json"), ("All files", "*.*")]
            )

            if self.model_path:
                if self.model_path.endswith('.json'):
                    with open(self.model_path, 'r') as f:
                        config = json.load(f)

                    self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                    self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                    self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                    self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                    if self.architecture.get() == "Cascade Transformer":
                            self.num_nodes.set(config.get("num_nodes", self.num_nodes.get()))

                    self.architecture.set(config.get("architecture", self.architecture.get()))
                    messagebox.showinfo("Model Loaded", f"Model configuration loaded from:\n{self.model_path}")

                elif self.model_path.endswith('.pth'):
                    config_path = os.path.join(os.path.dirname(self.model_path), 'model_config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)

                        self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                        self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                        self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                        self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                        self.architecture.set(config.get("architecture", self.architecture.get()))
                        if self.architecture.get() == "Cascade Transformer":
                            self.num_nodes.set(config.get("num_nodes", self.num_nodes.get()))

                        self.load_model()

                        state_dict = torch.load(self.model_path, map_location=self.device)
                        self.model.load_state_dict(state_dict, strict=False)
                        messagebox.showinfo("Success", f"Model weights loaded from:\n{self.model_path}")
                    else:
                        messagebox.showwarning(
                            title="Missing Config",
                            message="Configuration file not found.\nEnsure model_config.json exists in the same directory."
                        )
                else:
                    messagebox.showerror("Unsupported Format", "Unsupported file type selected.")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")


    def select_dataset(self):
        self.dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if self.dataset_path:
            messagebox.showinfo("Success", f"Dataset directory selected: {self.dataset_path}")

    def select_vocab(self):
        self.vocab_path = filedialog.askopenfilename(
            title="Select Vocabulary File",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if self.vocab_path:
            messagebox.showinfo("Success", f"Vocabulary file selected: {self.vocab_path}")

    def select_tokenizer(self):
        try:
    
            self.tokenizer_path = filedialog.askdirectory(title="Select Tokenizer Directory")

            if self.tokenizer_path:
                messagebox.showinfo("Success", f"Tokenizer file selected: {self.tokenizer_path}")

            # Updated tokenizer instantiation using from_pretrained:
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)
            # Log the vocabulary size.
            tokenizer_vocab_size = len(self.tokenizer)
            logging.info(f"Loaded tokenizer vocabulary size: {tokenizer_vocab_size}")
            self.vocab_size.set(tokenizer_vocab_size)
            print("Voca_size", self.vocab_size.get())
            print("Special tokens map:", self.tokenizer.special_tokens_map)
            print("Pad token ID:", self.tokenizer.pad_token_id)
            print("Model max length:", self.tokenizer.model_max_length)

            logging.info(f"Loaded tokenizer with pad_token: {self.tokenizer.pad_token} and pad_token_id: {self.tokenizer.pad_token_id}")

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")        

    def test_tokenizer(self):
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer not loaded.")
            return
        sample_text = simpledialog.askstring("Test Tokenizer", "Enter a sample text to tokenize:")
        if sample_text:
            tokens = self.tokenizer.tokenize(sample_text)
            token_ids = self.tokenizer.encode(sample_text)
            logging.info(f"Sample Text: {sample_text}")
            logging.info(f"Tokens: {tokens}")
            logging.info(f"Token IDs: {token_ids}")
            messagebox.showinfo("Tokenizer Test", f"Tokens: {tokens}\nToken IDs: {token_ids}")

    def save_dataset_as_text(self):
        if not hasattr(self, 'text_data') or not self.text_data:
            messagebox.showerror("Error", "No dataset loaded or processed to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Dataset as Text File",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    for line in self.text_data:
                        f.write(line + '\n')
                messagebox.showinfo("Success", f"Dataset saved to {save_path}")
                logging.info(f"Dataset saved to {save_path}")
            except Exception as e:
                logging.error(f"Failed to save dataset: {e}")
                messagebox.showerror("Error", f"Failed to save dataset: {e}")
                
    def create_tokenizer_from_vocab(self):
        try:
            # Ask the user to select the vocabulary file (our generated tokenizer.json)
            vocab_path = filedialog.askopenfilename(
                title="Select Vocabulary File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not vocab_path:
                messagebox.showerror("Error", "No vocabulary file selected.")
                return

            # Load the vocab from the JSON.
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            if "token_to_id" not in vocab_data:
                raise ValueError("The JSON file does not contain a 'token_to_id' key.")

            vocab = vocab_data["token_to_id"]

            # Check if merges exist in the file.
            if "merges" in vocab_data:
                merges = vocab_data["merges"]
                # Create a BPE model if merges are available.
                model = models.BPE(vocab=vocab, merges=merges, unk_token="<UNK>")
            else:
                # Fallback: use a WordLevel model if no merges are found.
                model = models.WordLevel(vocab=vocab, unk_token="<UNK>")

            # Create the tokenizer with the selected model.
            tokenizer = Tokenizer(model)

            # Set normalizer to NFKC for Unicode normalization.
            tokenizer.normalizer = normalizers.NFKC()
            # Use ByteLevel pre-tokenizer for byte-level tokenization.
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            # Use ByteLevel decoder for correct reconstruction of text.
            tokenizer.decoder = decoders.ByteLevel()

            # Wrap with PreTrainedTokenizerFast for HF integration.
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                unk_token="<UNK>",
                pad_token="<PAD>",
                bos_token="<BOS>",
                eos_token="<EOS>",
                model_max_length=seq_len  # Ensure seq_len is defined in your context.
            )

            # Ensure special tokens are added.
            self.tokenizer.add_special_tokens({
                'unk_token': '<UNK>',
                'pad_token': '<PAD>',
                'bos_token': '<BOS>',
                'eos_token': '<EOS>'
            })

            # Save the tokenizer.
            save_directory = filedialog.askdirectory(title="Select Directory to Save Tokenizer")
            if save_directory:
                os.makedirs(save_directory, exist_ok=True)
                self.tokenizer.save_pretrained(save_directory)
                self.tokenizer_path = os.path.join(save_directory, 'tokenizer.json')
                messagebox.showinfo("Success", f"Tokenizer saved to {self.tokenizer_path}")
                logging.info(f"Tokenizer saved to {self.tokenizer_path}")
            else:
                messagebox.showerror("Error", "No save directory selected for tokenizer.")
                return

            # Test the tokenizer.
            test_text = "Hello World!\nThis is a test.\tLet's remove line breaks and tabs."
            tokens = self.tokenizer.tokenize(test_text)
            logging.info(f"Test tokenization of '{test_text}': {tokens}")

            tokenizer_vocab = self.tokenizer.get_vocab()
            sorted_vocab = dict(sorted(tokenizer_vocab.items(), key=lambda item: item[1]))
            logging.info(f"Sorted Tokenizer Vocabulary: {sorted_vocab}")

            logging.info("Tokenizer created and saved successfully")
        except Exception as e:
            logging.error(f"Failed to create tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to create tokenizer: {str(e)}")
            raise


    def load_model(self):
        try:
            if not self.tokenizer:
                messagebox.showerror("Error! Select tokenizer first")
                return
            else:
                vocab_size = len(self.tokenizer)

            # Log and validate vocab size
            logging.info(f"Tokenizer vocabulary size: {vocab_size}")
            self.vocab_size.set(vocab_size)

            # Initialize the model based on architecture
            if self.architecture.get() == "Reasoning Model":
                self.model = Transformer_Model(
                    vocab_size=vocab_size,
                    embedding_dim=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),
                    num_heads=self.num_heads.get(),
                    tokenizer=self.tokenizer,
                    seq_length=seq_len
                )

            elif self.architecture.get() == "Cascade Transformer":
                self.model = CascadeTransformer(
    
                    vocab_size=vocab_size,
                    embed_size=self.hidden_size.get(),
                    hidden_size=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),
                    num_heads=self.num_heads.get(),
                    num_nodes=self.num_nodes.get(),
                    tokenizer=self.tokenizer,
                    max_seq_length=seq_len
                )
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            # Move the entire model to the selected device
            self.model.to(device)
            logging.info(f"Model moved to device: {self.device}")

            # Load checkpoint if a model file is selected
            if self.model_path and self.model_path.endswith('.pth'):
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)
                logging.info("Model weights loaded and resized successfully.")

            logging.info(f"Model initialized on device: {self.device}")
            messagebox.showinfo("Success", "Model initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")


    def calculate_learning_rate(self, total_params):
        # Calculate learning rate based on total parameters using the derived formula
        # LR = 17.38 * (Model Size)^-0.424
        lr = 17.38 * (total_params ** -0.424)
        return lr

    def start_training(self):
        # Start training in a separate thread to keep the GUI responsive
        self.stop_training.clear()
        training_thread = threading.Thread(target=self.training_loop)
        training_thread.start()

    def update_progress(self, progress_value):
        self.progress_bar['value'] = progress_value

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def save_checkpoint(self, model, optimizer, epoch, path):
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"Expected path to be str or os.PathLike, got {type(path).__name__}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        

    def validate_training_parameters(self):
        # Validate batch size
        try:
            batch_size = int(self.batch_size.get())
            if batch_size <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid batch size: {self.batch_size.get()}")
            messagebox.showerror("Error", "Batch size must be a positive integer.")
            return False

        # Validate epochs
        try:
            epochs = int(self.epochs.get())
            if epochs <= 0:
                raise ValueError
        except (TypeError, ValueError):
            logging.error(f"Invalid epochs value: {self.epochs.get()}")
            messagebox.showerror("Error", "Epochs must be a positive integer.")
            return False


        if not hasattr(self.tokenizer, 'pad_token_id') or self.tokenizer.pad_token_id is None:
            logging.error("Tokenizer pad_token_id is not set.")
            messagebox.showerror("Error", "Tokenizer is missing pad_token_id.")
            return False

        return True

    def training_loop(self):
        if not self.validate_training_parameters():
            return

        logging.info("All training parameters and data are properly initialized.")
        if not self.model:
            logging.error("Model not initialized before training")
            return
        self.use_genetic_algo = self.genetic_algo_var.get()

        try:


                # Initialize the standard dataset and dataloader
            device_cpu = 'cpu'
                # Ensure the tokenizer is loaded and has a valid pad_token_id
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer else pad_token_id  # Default to global if tokenizer isn't set      
            max_length = seq_len  # Adjust as needed
            logging.info("max_length set")

            logging.info("datas stacked and torched")

            dataset = RawPairDataset(self.query_target_pairs)

            # After tokenizer is loaded
            init_collate_globals(self.tokenizer)
            # In your GUI class, you then create your DataLoader with:

            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=int(self.batch_size.get()),
                    shuffle=False,
                    num_workers=32,    # Uses multiple CPU threads for data loading
                    pin_memory=True,
                    collate_fn=collate_fn_global                  
                )
            logging.info("dataloader defined")


            ##chunked vs. standard else complete

            # Adjust learning rate based on architecture
            total_params = self.num_parameters.get()
            lr = self.learning_rate.get()
            logging.info(f"Learning Rate: {lr} for total parameters: {total_params}")

            # Learning rate scheduler
            total_steps = self.epochs.get() * (len(dataset)/self.batch_size.get())
            logging.info(f"Total training steps: {total_steps}")
            # Separate parameters based on their shape.

            # Create two optimizers:
            #Enable for standard optimizer/scheduler
            #num_warmup_steps = total_steps // 10  # Warmup for 10% of training
            #scheduler = self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

            #optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            logging.info("Optimizers defined")

            #loss_fn = nn.CrossEntropyLoss(label_smoothing=0.01, ignore_index=pad_token_id)
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

            self.model.train()
            logging.info("Model set to training mode")
            progress_step = 0
            n = 0
            step_count = 0

            for epoch in range(self.epochs.get()):
                if self.stop_training.is_set():
                    logging.info("Training stopped by user.")
                    messagebox.showinfo("Info", "Training stopped by user.")
                    break
                optimizer.zero_grad()
                logging.debug("Optimizer gradients zeroed")
                epoch_loss = 0
                logging.info(f"Epoch {epoch+1} started")
                torch.cuda.empty_cache()

                # Training loop
                for batch_idx, (full_tokens) in enumerate(dataloader):
                    if self.stop_training.is_set():
                            logging.info("Training stopped by user.")
                            messagebox.showinfo("Info", "Training stopped by user.")
                            return

                        # Move batches and targets to the correct device 
                    full_tokens = full_tokens.to(device, non_blocking=True)

                    max_len = full_tokens.shape[1]
                        # Logging epoch and batch info
                    logging.debug(f'Batch shape: {max_len}')  # (batch_size, seq_len)
                    logging.debug(f'Using device: {self.device}')

                    architecture = self.architecture.get()
                    step_count = 0
                    total_loss = 0
                    pad_token_id = self.tokenizer.pad_token_id or 0

                    logging.debug(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}')

                        # Move batches and targets to the correct device 

                        # Logging epoch and batch info
                    logging.debug(f'Using device: {self.device}')

                    optimizer.zero_grad()
                    batch_loss = 0
                    step_count = 0

                    optimizer.zero_grad()
                    batch_loss = 0
                    step_count = 0
                    full_tokens = full_tokens.to(device, non_blocking=True)
                    #max_len = full_tokens.shape[1]

                    logging.debug(f'Batch shape: {max_len}')  # (batch_size, seq_len)

                    architecture = self.architecture.get()
                    for t in range(seq_len, max_len):
                        if self.stop_training.is_set():
                                logging.info("Training stopped by user.")
                                messagebox.showinfo("Info", "Training stopped by user.")
                                return
                        start = max(0, t - seq_len)
                        src = full_tokens[:, start:t]                # decoder input
                        tgt_ids = full_tokens[:, start + 1:t + 1]    # target input
                        logging.debug(f'tgt_ids: {tgt_ids}')  # (batch_size, seq_len)

                        # Clip if lengths don’t match due to short edges
                        min_len = min(src.size(1), tgt_ids.size(1))
                        src = src[:, -min_len:]
                        tgt_ids = tgt_ids[:, -min_len:]
                        logging.debug(f'tgt_ids after adjustment: {tgt_ids}')  # (batch_size, seq_len)

                        if src.size(1) == 0 or tgt_ids.size(1) == 0:
                            continue

                        active_mask = (tgt_ids[:, -1] != pad_token_id)
                        if active_mask.sum().item() == 0:
                            continue
                        logging.debug(f'active mask set: {active_mask}')  # (batch_size, seq_len)

                        targets_flat = tgt_ids.reshape(-1)                                    # [B*T]
                        active_mask_flat = (targets_flat != pad_token_id)                     # [B*T]
                        targets_filtered = targets_flat[active_mask_flat]                    # [N]
                        logging.debug(f"target reshaped Shape: {targets_filtered.shape}")

                        with torch.amp.autocast(device, dtype=torch.float16):  # Enable mixed precision

                                # Check the flag and run evolution once per epoch if requested:
                            if self.use_genetic_algo == "Genetic Algorithm":
                                    logging.info("Applying genetic algorithm evolution step...")
                                    qga = GeneticAlgorithm(self.model, lr)
                                    # Evolve using the same loss function and dataloader (or a validation subset)
                                    self.model = qga.evolve(loss_fn, src, targets_filtered, tgt_ids, architecture)
                                    gc.collect()
                                    torch.cuda.empty_cache()


                            else:
                                            
                                def forward_fn(src, tgt):
                                    return self.model(src, tgt, mode="train")
                                    # Forward pass
                                try:
                                        if architecture == "Cascade Transformer":

                                            logits = self.model(src, tgt_ids)

                                        else:
                                            logits = cp.checkpoint(forward_fn, src, tgt_ids, use_reentrant=False)
                                except Exception as e:
                                            raise ValueError(f"forward pass failed for {str(e)}")

                                # Reshape to [batch * seq_len, vocab] and filter by mask
                                logits_flat = logits.reshape(-1, logits.shape[-1])                    # [B*T, V]
                                logits_filtered = logits_flat[active_mask_flat]                      # [N, V]

                                if logits_filtered.size(0) == 0:
                                    continue  # skip if nothing to train on this step
                                logits_filtered = torch.nan_to_num(logits_filtered, nan=0.0, posinf=10.0, neginf=-10.0)


                                logging.debug(f"Shape of logits: {logits.shape}")
                                    # Assume batch_labels are tensors of shape [batch_size, seq_len, vocab_size]
                                logging.debug(f"logits reshaped Shape: {logits_filtered.shape}")

                                step_loss = loss_fn(logits_filtered, targets_filtered)
                                if torch.isnan(step_loss).any():
                                    logging.warning("⚠️ Skipping optimizer step due to NaN loss.")
                                if not torch.isfinite(step_loss):
                                    print("🚨 Warning: NaN or Inf detected in loss. Skipping update.")
                                    optimizer.zero_grad()
                                    continue
                                step_loss.backward(retain_graph=True)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                logging.info(f"Loss computed: {step_loss.item()}")
                                # ✅ Check for NaNs in loss

                                # Backward pass and optimization

                                logging.info("Loss backward computed")
                                n+=1
                                print(f"Iteration {n}, step: {t}, Loss: {step_loss.item()}")
                                src = src.detach()
                                tgt_ids = tgt_ids.detach()
                                for name, param in self.model.named_parameters():
                                    if param.grad is not None and not torch.isfinite(param.grad).all():
                                        print(f"🚨 Non-finite grad in {name}, skipping step")
                                        optimizer.zero_grad()
                                        continue
                                optimizer.step()
                                gc.collect()
                                torch.cuda.empty_cache()

                                #print(f"  💥 Loss: {step_loss.item():.4f}")
                                #print(f"  🧠 GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

                                batch_loss += step_loss.item()
                                step_count += 1

                                if step_count > 0:
                                    if self.stop_training.is_set():
                                        logging.info("Training stopped by user.")
                                        messagebox.showinfo("Info", "Training stopped by user.")
                                        break
                                        # Check for NaN or Inf in gradients
                                    for name, param in self.model.named_parameters():
                                            if param.grad is not None:
                                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                                    logging.error(f"Gradient for {name} contains NaN or Inf.")
                                                
                                    for name, param in self.model.named_parameters():
                                        if param.grad is not None:
                                                logging.debug(f"Gradient for {name}: mean={param.grad.mean().item():.4f}, max={param.grad.max().item():.4f}, min={param.grad.min().item():.4f}")
                                        else:
                                                logging.debug(f"Gradient for {name} is None")

                                    total_norm = 0.0
                                    for p in self.model.parameters():
                                            if p.grad is not None:
                                                total_norm += p.grad.data.norm(2).item() ** 2
                                    total_norm = total_norm ** 0.5
                                    logging.info(f"Gradient norm: {total_norm}")

                                    for name, param in self.model.named_parameters():
                                        if param.grad is not None:
                                            logging.debug(f"Gradients for {name}: {param.grad}")
                                        else:
                                            logging.debug(f"No gradients found for {name}.")
                                    
                                        # Before optimizer step
                                    for name, param in self.model.named_parameters():
                                            if param.requires_grad:
                                                logging.debug(f"Before step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                                            # Log gradients for debugging
                                    optimizer.step()
                                    logging.info("Optimizer step update completed")

                                        # After optimizer step
                                    for name, param in self.model.named_parameters():
                                        if param.requires_grad:
                                            logging.debug(f"After step - {name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")
                                    optimizer.zero_grad()
                                    logging.debug("Optimizer gradients zeroed")
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    avg_loss = batch_loss / step_count
                    #print(f"📦 Batch {batch_idx // batch_size + 1}: Avg loss {avg_loss:.4f} over {step_count} steps")

                    n+=1
                    print(f"Batch iteration {batch_idx+1}, Loss: {avg_loss}")
                    gc.collect()
                    torch.cuda.empty_cache()

                                                    
                    progress_step += 1
                    progress_value = (progress_step / total_steps) * 100
                    self.root.after(0, self.update_progress, progress_value)

                # Log epoch loss
                logging.info(f"Epoch {epoch + 1}/{self.epochs.get()} completed")
                self.root.after(0, self.update_status, f"Epoch {epoch + 1}/{self.epochs.get()} completed.")

        except Exception as e:
            logging.error(f"An error occurred during training: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during training: {str(e)}")


    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
            
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def save_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Cannot save.")
            logging.error("Attempted to save model but model was not initialized.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Cannot save.")
            logging.error("Attempted to save model but tokenizer was not initialized.")
            return
        architecture = self.architecture.get()
        save_directory = filedialog.askdirectory(title="Select Save Directory")
        if save_directory:
            if architecture == "Cascade Transformer":

                config = {
                "vocab_size": self.vocab_size.get(),
                "embed_size": self.hidden_size.get(),
                "hidden_size": self.hidden_size.get(),
                "num_layers": self.num_layers.get(),
                "architecture": self.architecture.get(),
                "num_parameters": self.num_parameters.get(),
                "num_heads": self.num_heads.get(),
                "num_nodes": self.num_nodes.get(),
                "layers": self.layers
            }
            else:
                config = {
                "vocab_size": self.vocab_size.get(),
                "embed_size": self.hidden_size.get(),
                "hidden_size": self.hidden_size.get(),
                "num_layers": self.num_layers.get(),
                "architecture": self.architecture.get(),
                "num_parameters": self.num_parameters.get(),
                "num_heads": self.num_heads.get(),
                "layers": self.layers
            }
            config_path = os.path.join(save_directory, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f)

            # Ensure embeddings match tokenizer
            tokenizer_vocab_size = len(self.tokenizer)

            # Save the model state dictionary
            if self.architecture.get() == "Reasoning Model":
                model_file_name = 'reasoning_model.pth'
            elif self.architecture.get() == "Cascade Transformer":
                model_file_name = 'cascade_transformer.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(save_directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save the tokenizer
            self.tokenizer.save_pretrained(save_directory)

            messagebox.showinfo("Success", "Model, tokenizer, and config saved successfully.")
            logging.info("Model, tokenizer, and config saved successfully.")

    def stop_training_command(self):
        self.stop_training.set()
        messagebox.showinfo("Stop Training", "Training stopped.")
        logging.info("Training stopped by user.")

    def expand_transformer(self):
        # Placeholder method; not used in current implementation
        pass

    
    def load_dataset(self):
            # Load standard dataset
            if not self.dataset_path:
                messagebox.showerror("Error", "No dataset directory selected.")
                return

            dataset_files = os.listdir(self.dataset_path)
            self.query_target_pairs = []

            for file in dataset_files:
                file_path = os.path.join(self.dataset_path, file)
                if file.endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        text_data = list
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                        else:
                            messagebox.showerror(
                                "Error", f"CSV file '{file}' missing 'text' or 'instruct' and 'output' columns."
                            )
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read CSV file '{file}': {str(e)}")
                elif file.endswith('.json'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if file.endswith('.jsonl'):
                                for line in f:
                                    conversation = json.loads(line.strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                            else:
                                data = json.load(f)
                                self.query_target_pairs.extend(self.extract_query_target_pairs(data)) 
                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                               
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read JSON file '{file}': {str(e)}")
                elif file.endswith('.parquet'):
                    try:
                        df = pd.read_parquet(file_path)
                        if 'text' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['text'].strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                        elif 'TEXT' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['TEXT'].strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                        elif 'messages' in df.columns:
                                for df in df.columns:
                                    conversation = json.loads(df['messages'].strip())
                                    self.query_target_pairs.extend(self.extract_query_target_pairs([conversation]))

                                # After loading query_target_pairs
                                for i in range(min(5, len(self.query_target_pairs))):
                                    query, target = self.query_target_pairs[i]
                        elif 'instruct' in df.columns and 'output' in df.columns:
                            # Handle 'instruct' and 'output' columns
                            df = df.dropna(subset=['instruct', 'output'])
                            query = df['instruct'].astype(str).tolist()
                            target = df['output'].astype(str).tolist()
                        else:
                            messagebox.showerror(
                                "Error", f"Parquet file '{file}' missing 'text' or 'instruct' and 'output' columns."
                            )
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read Parquet file '{file}': {str(e)}")
                
                elif file.endswith('.txt'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        text_data.append(text)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read text file '{file}': {str(e)}")
                elif file.endswith('.pdf'):
                    try:
                        text = []
                        text = extract_text_from_pdf(file_path)
                        
                        # Break into query/target pairs
                        data = []
                        for i in range(0, len(text) - seq_len, seq_len):
                            query = text[i:i + seq_len]
                            target = text[i + 1:i + seq_len + 1]
                            self.query_target_pairs.append({query, target})
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read text file '{file}': {str(e)}")
                else:
                    messagebox.showwarning("Warning", f"Unsupported file format: '{file}'")

            if not self.query_target_pairs:
                messagebox.showerror("Error", "No valid query/target pairs found in the dataset.")
                return

            # Store text data for saving as a text file
            self.text_data = []
            for query, target in self.query_target_pairs:
                self.text_data.append(f"User: {query}\nAssistant: {target}")

            messagebox.showinfo("Success", f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")
            logging.info(f"Loaded dataset with {len(self.query_target_pairs)} query/target pairs.")


    def extract_query_target_pairs(self, data):
        query_target_pairs = []

        for conversation in data:
            if conversation.get("messages"):
                messages = conversation.get("messages", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                        query = messages[i].get("content") or messages[i].get("value", "")
                        target = messages[i + 1].get("content") or messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

                    elif messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))

            elif conversation.get("conversations"):
                messages = conversation.get("conversations", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            elif conversation.get("text"):
                messages = conversation.get("text", [])
                for i in range(len(messages) - 1):
                    if messages[i].get("from") == "user" and messages[i + 1].get("from") == "assistant":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
                    elif messages[i].get("from") == "human" and messages[i + 1].get("from") == "gpt":
                        query = messages[i].get("value", "")
                        target = messages[i + 1].get("value", "")
                        query_target_pairs.append((query.strip(), target.strip()))
            else:
                user_messages = conversation.get("user", [])
                assistant_messages = conversation.get("assistant", [])
                for i in range(min(len(user_messages), len(assistant_messages))):
                    query = user_messages[i].replace('\n', ' ').strip()
                    target = assistant_messages[i].replace('\n', ' ').strip()
                    query_target_pairs.append((query, target))
            # Final fallback: split everything into sequence-length chunks for predictive text
            if not query_target_pairs:
                all_text = " ".join([m.get("text", "") for conversation in data for m in conversation])
                tokenized_text = self.tokenizer.encode(all_text, truncation=False)
                query_target_pairs = [
                    {"query": tokenized_text[i:i+seq_len], "target": tokenized_text[i:i+seq_len]}
                    for i in range(0, len(tokenized_text), seq_len)
                ]

        return query_target_pairs

    def extract_query_target_pairs_json(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                f.seek(0)  # Reset file pointer
                
                if first_line.startswith('['):
                    data = json.load(f)  # JSON array format
                else:
                    data = [json.loads(line.strip()) for line in f]  # JSONL format

            return self.extract_query_target_pairs(data)

        except Exception as e:
            logging.error(f"Failed to load JSON file: {e}")
            return []


    def extract_query_target_pairs_parquet(self, file_path):
        try:
            df = pd.read_parquet(file_path)
            query_target_pairs = []

            for _, row in df.iterrows():
                user_query = row.get("question") or row.get("input")
                assistant_response = row.get("answer") or row.get("response")

                if user_query and assistant_response:
                    query_target_pairs.append((user_query.strip(), assistant_response.strip()))

            return query_target_pairs

        except Exception as e:
            logging.error(f"Failed to load Parquet file: {e}")
            return []


# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()


    app = ReasoningModelGUI(root)
    root.mainloop()
