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
#torch.set_num_threads(32)
#torch.set_num_interop_threads(32)

# Debug for CUDA

tokenizer = None
# Debug for CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_LOGINFO_DBG"]= "0"
os.environ["CUBLAS_LOG_LEVEL"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
tokenizer = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

seq_len = 256
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


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    logging.info(f"Tokenizer pad_token set to: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")


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

def collate_fn(batch):
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

class LogarithmicNumberSystem:
    """
    CUDA-Optimized Logarithmic Number System (LNS) for efficient GPU computation.
    """
    def __init__(self, epsilon=1e-6, use_cuda=True):
        self.epsilon = epsilon
        self.device = device

    def to_log(self, x):
        logging.debug(f"shape of x to log {x.shape}")

        """ Convert tensor to log-space using CUDA acceleration. """
        return torch.log(torch.clamp(x.to(self.device), min=self.epsilon))

    def from_log(self, log_x, epsilon=1e-30):
        logging.debug(f"shape of log_x to to convert back {log_x.shape}")
        if torch.isnan(log_x).any() or torch.isinf(log_x).any():
            logging.error("NaN or Inf detected in from log input!")
            logging.error(f"log_x Projection: {log_x}")        # Add epsilon to prevent exp(-inf) = 0
        return torch.exp(log_x) + epsilon


    def log_add(self, log_x, log_y):
        """ Logarithmic addition using CUDA-accelerated Log-Sum-Exp trick. """
        logging.debug(f"shape of log_x for add {log_x.shape}")
        logging.debug(f"shape of log_y for add {log_y.shape}")

        max_val, _ = torch.max(torch.stack([log_x, log_y], dim=-1), dim=-1)
        return max_val + torch.log(torch.exp(log_x - max_val) + torch.exp(log_y - max_val))

    def log_sub(self, log_x, log_y):
        """ Logarithmic subtraction with CUDA support. """
        logging.debug(f"shape of log_x for sub {log_x.shape}")
        logging.debug(f"shape of log_y for sub {log_y.shape}")
        max_val, _ = torch.max(torch.stack([log_x, log_y], dim=-1), dim=-1)
        logging.debug(f"shape of max_val for sub {log_x.shape}")
        sub_result = torch.exp(log_x - max_val) - torch.exp(log_y - max_val)
        logging.debug(f"shape of sub_result for sub {log_x.shape}")
        
        return max_val + torch.log(torch.clamp(sub_result, min=self.epsilon))

    def log_mul(self, log_x, log_y):
        """ Logarithmic multiplication using CUDA (log-space addition). """
        logging.debug(f"shape of log_x for mul {log_x.shape}")
        logging.debug(f"shape of log_y for mul {log_y.shape}")
        return log_x + log_y

    def log_matmul(self, log_A, log_B):
        """
        Log-space matrix multiplication.
        Computes log_C = log_sum_exp(log_A + log_B) across the last dim of A and first dim of B.
        """
        # Expand dimensions to make them broadcastable for addition
        log_A = log_A.unsqueeze(-1)  # Shape: (..., M, N, 1)
        log_B = log_B.unsqueeze(-3)  # Shape: (..., 1, N, P)
        
        # Add and sum across the shared dimension
        log_product = log_A + log_B  # Shape: (..., M, N, P)
        return torch.logsumexp(log_product, dim=-2)  # Sum across the shared dimension

    def log_div(self, log_x, log_y):
        """ Logarithmic division using CUDA (log-space subtraction). """
        logging.debug(f"shape of log_x for div {log_x.shape}")
        logging.debug(f"shape of log_y for div {log_y.shape}")
        return log_x - log_y
    
    def log_add_einsum(self, equation, log_x, log_y):
        """
        Implements log-space einsum operation by applying log-sum-exp trick.
        """
        # Ensure tensors have same shape
        assert log_x.shape == log_y.shape, f"Shape mismatch: {log_x.shape} vs {log_y.shape}"

        max_val = torch.max(log_x, log_y)
        logging.debug(f"shape of max_val for einsum {max_val.shape}")
        logging.debug(f"shape of log_x for einsum {log_x.shape}")
        logging.debug(f"shape of log_y for einsum {log_y.shape}")
        log_x_adj = log_x - max_val
        log_y_adj = log_y - max_val
        logging.debug(f"Einsum equation: {equation}")
        logging.debug(f"log_x_adj shape: {log_x_adj.shape}")
        logging.debug(f"log_y_adj shape: {log_y_adj.shape}")
        log_x_adj = log_sum_exp(log_x_adj, dim=-1)
        #log_x_adj = log_x_adj.expand(-1,-1,128, -1)
        log_y_adj = log_sum_exp(log_y_adj, dim=-1)
        #log_y_adj = log_y_adj.expand(-1,-1,128, -1)
        logging.debug(f"log_x_adj shape after log_sum_exp: {log_x_adj.shape}")
        logging.debug(f"log_y_adj shape after log_sum_exp: {log_y_adj.shape}")
        einsum_tensor = torch.einsum(equation, [log_x_adj, log_y_adj])
        logging.debug(f"einsum_tenspr shape: {einsum_tensor.shape}")
        einsum_tensor = einsum_tensor.unsqueeze(-1)
        # ✅ Ensure max_val reduces along the last dim before logsumexp
        max_val, _ = torch.max(einsum_tensor, dim=-1, keepdim=True)  
        logging.debug(f"Shape of max_val: {max_val.shape}")  # Should be [batch, seq_len, seq_len, 1]
        einsum_tensor_adj = einsum_tensor - max_val

        logging.debug(f"Shape of einsum t after max subtraction: {einsum_tensor_adj.shape}")
        einsum_tensor_adj = torch.logsumexp(einsum_tensor_adj, dim=-1)
        logging.debug(f"Shape einsum t before sum: {einsum_tensor_adj.shape}")
        # ✅ Apply logsumexp only across the correct dimension

        return  einsum_tensor_adj


def log_sum_exp(tensor, dim=-1, keepdim=True):
    """
    Optimized Log-Sum-Exp function for stable summation in log-space.
    Prevents overflow and underflow issues by normalizing.
    """
    logging.debug(f"shape of tensor for log_sum_exp {tensor.shape}")
    
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)  # Find max value
    return max_val + torch.log(torch.sum(torch.exp(tensor - max_val), dim=dim, keepdim=keepdim))

##change to log space
class LinearLNS(nn.Module):
    r"""Applies an logarithmic linear transformation to the incoming data

    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.lns = LogarithmicNumberSystem()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logging.debug(f"input shape ff: {input.shape}")
        log = self.lns.to_log(input.clone())
        logging.debug(f"log shape ff: {log.shape}")
        weight = self.lns.to_log(self.weight.clone())
        logging.debug(f"weight shape ff: {weight.shape}")
        bias = self.lns.to_log(self.bias.clone())
        logging.debug(f"bias shape ff: {bias.shape}")
        intermediate = self.lns.log_matmul(log, weight)
        logging.debug(f"intermediate shape ff: {intermediate.shape}")
        # Expand bias to match the shape of intermediate
        bias = bias.view(1, 1, -1).expand_as(intermediate)
        logging.debug(f"bias expanded shape ff: {bias.shape}")
        output = self.lns.log_add(intermediate, bias)
        logging.debug(f"output shape ff: {output.shape}")
        return self.lns.from_log(output)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

##change to log space
class EmbeddingLNS(nn.Module):
    r""" Modified for LNS, From torch.nn
    """

    __constants__ = [
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ]

    num_embeddings: int
    embedding_dim: int
    padding_idx: typing.Optional[int]
    max_norm: typing.Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: torch.Tensor
    freeze: bool
    sparse: bool

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: typing.Optional[int] = None,
        max_norm: typing.Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: typing.Optional[torch.Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                requires_grad=not _freeze,
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = nn.Parameter(_weight, requires_grad=not _freeze)

        self.sparse = sparse
        self.lns = LogarithmicNumberSystem()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(
        cls,
        embeddings,
        freeze=True,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        r"""Create Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (bool, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                         therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                         i.e. it remains as a fixed "pad".
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (bool, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert (
            embeddings.dim() == 2
        ), "Embeddings parameter is expected to be 2-dimensional"
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )
        return embedding


class TransformerEncoderLayerLNS(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayerLNS, self).__init__()

        self.self_attn = MultiheadAttentionLNS(d_model, num_heads, dropout)
        self.ffn = FeedForwardLNS(d_model, dim_feedforward)

        # Layer normalization and dropout in log space
        self.norm1 = LayerNormLNS(d_model)
        self.norm2 = LayerNormLNS(d_model)
        self.dropout1 = DropoutLNS(dropout)
        self.dropout2 = DropoutLNS(dropout)
        self.lns = LogarithmicNumberSystem()

    def forward(self, src, src_mask=None):
        # Self-attention in log space
        log_attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        logging.debug(f"attn_output shape: {log_attn_output.shape}")
        log_src = self.lns.to_log(src.clone())
        # Residual connection and layer normalization in log space
        log_src = self.lns.log_add(log_src, self.lns.to_log(self.dropout1(log_attn_output)))
        log_src = self.lns.from_log(log_src.clone())
        log_src = self.norm1(log_src)
        logging.debug(f"src shape after norm1: {log_src.shape}")
        
        # Feed-forward network in log space
        log_ff_output = self.ffn(log_src)
        logging.debug(f"ff_output shape: {log_ff_output.shape}")
        
        # Residual connection and layer normalization in log space
        log_src = self.lns.log_add(self.lns.to_log(log_src), self.lns.to_log(self.dropout2(log_ff_output)))
        log_src = self.norm2(self.lns.from_log(log_src))
        
        return log_src

class TransformerDecoderLayerLNS(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerDecoderLayerLNS, self).__init__()

        # Self-attention and cross-attention layers using MultiheadAttentionLNS
        self.self_attn = MultiheadAttentionLNS(d_model, num_heads, dropout)
        self.multihead_attn = MultiheadAttentionLNS(d_model, num_heads, dropout)

        # Feed-forward layers using FeedForwardLNS
        self.ffn = FeedForwardLNS(d_model, dim_feedforward)

        # Layer normalization and dropout in log space
        self.norm1 = LayerNormLNS(d_model)
        self.norm2 = LayerNormLNS(d_model)
        self.norm3 = LayerNormLNS(d_model)
        self.dropout1 = DropoutLNS(dropout)
        self.dropout2 = DropoutLNS(dropout)
        self.dropout3 = DropoutLNS(dropout)
        
        self.lns = LogarithmicNumberSystem()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention for target in log space
        log_attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        log_tgt = self.lns.to_log(tgt.clone())
        log_tgt = self.lns.log_add(log_tgt, self.lns.to_log(self.dropout1(log_attn_output)))
        log_tgt = self.norm1(self.lns.from_log(log_tgt))
        logging.debug(f"Log_tgt shape after self-attn: {log_tgt.shape}")

        # Cross-attention with encoder output (memory) in log space
        log_cross_attn_output, _ = self.multihead_attn(log_tgt, memory, memory, attn_mask=memory_mask)
        log_tgt = self.lns.log_add(self.lns.to_log(log_tgt), self.lns.to_log(self.dropout2(log_cross_attn_output)))
        log_tgt = self.norm2(self.lns.from_log(log_tgt))
        logging.debug(f"Log_tgt shape after cross-attn: {log_tgt.shape}")
        
        # Feed-forward network in log space
        log_ff_output = self.ffn(log_tgt)
        log_tgt = self.lns.log_add(self.lns.to_log(log_tgt), self.lns.to_log(self.dropout3(log_ff_output)))
        log_tgt = self.norm3(self.lns.from_log(log_tgt))
        logging.debug(f"Log_tgt shape after FFN: {log_tgt.shape}")

        return log_tgt


class TransformerEncoderLNS(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerEncoderLNS, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerLNS(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNormLNS(d_model)
        self.lns = LogarithmicNumberSystem()

    def forward(self, src, src_mask=None):
        logging.debug(f"Log_src shape: {src.shape}")

        for layer in self.layers:
            log_src = layer(src, src_mask)
        return self.norm(log_src)


class TransformerDecoderLNS(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, num_layers, dropout=0.1):
        super(TransformerDecoderLNS, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayerLNS(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNormLNS(d_model)
        self.lns = LogarithmicNumberSystem()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        logging.debug(f"tgt shape: {tgt.shape}")

        logging.debug(f"memory shape: {memory.shape}")
        
        for layer in self.layers:
            log_tgt = layer(tgt, memory, tgt_mask, memory_mask)
        
        return self.norm(log_tgt)


class MultiheadAttentionLNS(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiheadAttentionLNS, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.lns = LogarithmicNumberSystem()
        
        # Correct the projections to project to num_heads * d_k
        self.q_proj = LinearLNS(d_model, num_heads * self.d_k)
        self.k_proj = LinearLNS(d_model, num_heads * self.d_k)
        self.v_proj = LinearLNS(d_model, num_heads * self.d_k)
        
        # Output projection to d_model
        self.out_proj = LinearLNS(num_heads * self.d_k, d_model)
        
        # Dropout
        self.dropout = DropoutLNS(dropout)
    
    def forward(self, query, key, value, attn_mask=None):
        """
        Forward pass for MultiHeadAttentionLNS.
        Performs attention calculations in normal space for stability.
        """
        batch_size = query.size(0)
        logging.debug(f"query shape multihead attention: {query.shape}")
        logging.debug(f"key shape multihead attention: {key.shape}")
        logging.debug(f"value shape multihead attention: {value.shape}")
        
        # 1. Linear projections for Q, K, V
        log_q = self.lns.to_log(self.q_proj(query))
        log_k = self.lns.to_log(self.k_proj(key))
        log_v = self.lns.to_log(self.v_proj(value))
        
        # 2. Split heads and rearrange dimensions
        log_q = log_q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        log_k = log_k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        log_v = log_v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        logging.debug(f"log_q shape multihead attention: {log_q.shape}")
        logging.debug(f"log_k shape multihead attention: {log_k.shape}")
        logging.debug(f"log_v shape multihead attention: {log_v.shape}")
        
        # === Perform Attention in Normal Space ===
        # Convert to normal space
        q = self.lns.from_log(log_q)
        k = self.lns.from_log(log_k)
        v = self.lns.from_log(log_v)
        logging.debug(f"q shape multihead attention: {q.shape}")
        logging.debug(f"k shape multihead attention: {k.shape}")
        logging.debug(f"v shape multihead attention: {v.shape}")
        
        # 3. Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
   
        if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
            logging.error("NaN or Inf detected in attention!")
            logging.error(f"attn_scores 1: {attn_scores}")
        logging.debug(f"attn_scores shape multihead attention1: {attn_scores.shape}")
    
        # 4. Apply attention mask
        if attn_mask is not None:
            # Ensure mask is broadcastable and uses -1e9 for softmax stability
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(1) == 0, -1e9)
            logging.debug(f"attn_mask unsqueezed shape multihead attention: {attn_mask.shape}")
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                logging.error("NaN or Inf detected in attention!")
                logging.error(f"attn_scores 1: {attn_scores}")
            logging.debug(f"attn_scores shape multihead attention2: {attn_scores.shape}")
            
        # 5. Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        logging.debug(f"attn_probs shape multihead attention: {attn_probs.shape}")
        if torch.isnan(attn_probs).any() or torch.isinf(attn_probs).any():
            logging.error("NaN or Inf detected in attention!")
            logging.error(f"attn_probs: {attn_probs}")

        # 6. Weighted sum of values
        weighted_values = torch.matmul(attn_probs, v)
        logging.debug(f"weighted_values shape multihead attention: {weighted_values.shape}")
        if torch.isnan(weighted_values).any() or torch.isinf(weighted_values).any():
            logging.error("NaN or Inf detected in attention!")
            logging.error(f"weighted values: {weighted_values}")
                    
        # 7. Convert back to log space
        log_attention_output = self.lns.to_log(weighted_values)
        logging.debug(f"log_attention_output shape multihead attention 1: {log_attention_output.shape}")
        if torch.isnan(log_attention_output).any() or torch.isinf(log_attention_output).any():
            logging.error("NaN or Inf detected in attention!")
            logging.error(f"log_attention_output 1: {log_attention_output}")
                    
        # 8. Combine heads into the last dimension
        log_attention_output = log_attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        logging.debug(f"log_attention_output shape multihead attention 2: {log_attention_output.shape}")
        if torch.isnan(log_attention_output).any() or torch.isinf(log_attention_output).any():
            logging.error("NaN or Inf detected in attention!")
            logging.error(f"log_attention_output 2: {log_attention_output}")
                    
        # 9. Output projection
        log_attention_output = self.lns.to_log(self.out_proj(self.lns.from_log(log_attention_output)))
        logging.debug(f"log_attention_output shape multihead attention 3: {log_attention_output.shape}")
        if torch.isnan(log_attention_output).any() or torch.isinf(log_attention_output).any():
            logging.error("NaN or Inf detected in attention!")
            logging.error(f"log_attention_output 3: {log_attention_output}")
                    
        # 10. Apply dropout and return
        log_attention_output = self.dropout(log_attention_output)
        logging.debug(f"log_attention_output shape multihead attention 4: {log_attention_output.shape}")
        if torch.isnan(log_attention_output).any() or torch.isinf(log_attention_output).any():
            logging.error("NaN or Inf detected in attention!")
            logging.error(f"log_attention_output 4: {log_attention_output}")
            
        return log_attention_output, attn_probs


    
class AttentionLNS(nn.Module):
    """
    Attention using CUDA-accelerated Logarithmic Number System.
    """
    def __init__(self, embedding_dim):
        super(AttentionLNS, self).__init__()
        self.lns = LogarithmicNumberSystem()
        self.query_weight = LinearLNS(embedding_dim, embedding_dim, seq_len)
        self.key_weight = LinearLNS(embedding_dim, embedding_dim, seq_len)
        self.value_weight = LinearLNS(embedding_dim, embedding_dim, seq_len)

    def forward(self, x, mask=None):
        # Compute standard linear projections.
        Q = self.query_weight(x)  # shape: [B, T, d]
        K = self.key_weight(x)    # shape: [B, T, d]
        V = self.value_weight(x)  # shape: [B, T, d]
        
        # Clamp to avoid zeros, then convert to log-space.
        Q_lin = torch.clamp(Q, min=1e-6)
        K_lin = torch.clamp(K, min=1e-6)
        logQ = self.lns.to_log(Q_lin)  # shape: [B, T, d]
        logK = self.lns.to_log(K_lin)  # shape: [B, T, d]
        
        # Compute dot product in log-space.
        # For each pair (i, j) over time, we want:
        #   log_score(i,j) = log(sum_k exp(logQ[i,k] + logK[j,k]))
        # We can use broadcasting: add logQ (unsqueezed) and logK (unsqueezed) then log_sum_exp.
        log_scores = log_sum_exp(logQ.unsqueeze(2) + logK.unsqueeze(1), dim=-1)
        
        # Scale by subtracting log(sqrt(d)) (i.e. equivalent to division in linear space).
        d = Q.size(-1)
        log_scale = math.log(math.sqrt(d))
        log_scores = log_scores - log_scale

        # If a mask is provided, mask out unwanted positions (assume mask==0 means masked).
        if mask is not None:
            log_scores = log_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute log-softmax: for each query, subtract log(sum(exp(log_scores)))
        log_sum = log_sum_exp(log_scores, dim=-1, keepdim=True)
        log_softmax = log_scores - log_sum
        
        # Convert back to linear probabilities.
        attention_weights = torch.exp(log_softmax)
        
        # Finally, compute attended output in the usual way.
        attended_output = torch.matmul(attention_weights, V)
        return attended_output



class FeedForwardLNS(nn.Module):
    """
    Optimized Quaternion Feed Forward Network using CUDA-accelerated Logarithmic Number System (LNS).
    """
    def __init__(self, embedding_dim, hidden_dim):
        super(FeedForwardLNS, self).__init__()
        self.lns = LogarithmicNumberSystem()
        self.fc1 = LinearLNS(embedding_dim, hidden_dim, seq_len)
        self.activation = nn.GELU()  
        self.fc2 = LinearLNS(hidden_dim, embedding_dim, seq_len)

    def forward(self, x):
        logging.debug(f"Feed-forward input shape: {x.shape}")

        log_hidden = self.lns.log_add(self.lns.to_log(self.fc1(x)), self.lns.to_log(self.activation(x)))
        logging.debug(f"Hidden layer output shape: {log_hidden.shape}")

        output = self.fc2(self.lns.from_log(log_hidden))
        logging.debug(f"Feed-forward output shape: {output.shape}")

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # Instead of erroring, simply truncate positional encodings to x.size(1)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class PositionalEncoding_LNS(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.dropout = DropoutLNS(dropout)
        self.lns = LogarithmicNumberSystem()  # Use the existing LNS class
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        
        # Compute positional encodings in log space
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        # Convert positional encodings to log space
        #self.log_pe = self.lns.to_log(pe)
        # Convert positional encodings to log space and register directly as a buffer
        self.register_buffer('log_pe', self.lns.to_log(pe))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        logging.debug(f"x Shape for pos encoder lns: {x.shape}") 

        # Get the relevant positional encodings in log space
        log_pos_encodings = self.log_pe[:, :x.size(1), :]  # Shape: (1, seq_len, d_model)

        # Expand to match batch size
        log_pos_encodings = log_pos_encodings.expand(x.size(0), -1, -1)  # Shape: (batch, seq_len, d_model)
        logging.debug(f"log_pos_encodings Shape pos encoder lns: {log_pos_encodings.shape}") 

        # Add positional encodings in log space
        log_x = self.lns.to_log(x)
        logging.debug(f"log_x Shape pos encoder lns: {log_x.shape}") 
        log_x = self.lns.log_add(log_x, log_pos_encodings)
        logging.debug(f"log_x 2 Shape: {log_x.shape}") 
        
        return log_x

### REASONING PIPELINE ###
class Transformer_Model_LNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, seq_length, num_heads):
        super(Transformer_Model_LNS, self).__init__()
        self.embedding = EmbeddingLNS(vocab_size, embedding_dim)
        self.output_projection = LinearLNS(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding_LNS(seq_length, embedding_dim)
        self.vocab_size=vocab_size
        self.transformer_encoder = TransformerEncoderLNS(embedding_dim, num_heads, hidden_dim, num_layers)
        self.lns = LogarithmicNumberSystem()
        self.decoder = TransformerDecoderLNS(
            d_model=embedding_dim, 
            num_heads=num_heads, 
            dim_feedforward=hidden_dim, 
            num_layers=num_layers
        )
        self.embedding_dim = embedding_dim 

    def forward(self, src, tgt_ids=None, mode='eval'):

        if isinstance(src[0], str):
            input_ids = self.tokenizer_wrapper.encode(src)
        if tgt_ids is not None and isinstance(tgt_ids[0], str):
            target= self.tokenizer_wrapper.encode(tgt_ids)
        elif tgt_ids is not None and mode == 'train':
            target = tgt_ids
            #tgt_ids = tgt_ids[:, 1:]
        #print(f"\n🚀 FORWARD: src shape {src.shape}, tgt shape {tgt_ids.shape}")
        elif tgt_ids is not None and tgt_ids.size(1) == 0:
            raise ValueError("❌ Decoder input has 0 length!")

        logging.debug(f"Input IDs Shape: {input_ids.shape}")  # Log input shape
        torch.autograd.set_detect_anomaly(True)

        x = self.embedding(input_ids)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("NaN or Inf detected in model!")
            logging.error(f"x: {x}")
        logging.debug(f"Embedding Output Shape: {x.shape}")  # Log embedding output
        tgt_emb = self.embedding(target) * math.sqrt(self.embedding_dim)
        logging.debug(f"tgt_emb Output Shape: {tgt_emb.shape}")  # Log embedding output
        
        tgt_emb = self.pos_encoder(tgt_emb)
        logging.debug(f"tgt_emb pos_encoder Shape: {tgt_emb.shape}") 
        if torch.isnan(tgt_emb).any() or torch.isinf(tgt_emb).any():
            logging.error("NaN or Inf detected in model!")
            logging.error(f"tgt_emb: {tgt_emb}")

        batch_size, seq_length, embedding_dim = x.size()
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).unsqueeze(-1)
        positions = positions.expand(batch_size, seq_length, embedding_dim)
        pos_encodings = self.pos_encoder(positions)  # Change to log space

        logging.debug(f"pos_encoding Shape1: {pos_encodings.shape}") 
        
        pos_encodings = pos_encodings.expand(batch_size, seq_length, -1)
        logging.debug(f"pos_encoding Shape2: {pos_encodings.shape}") 
        log_x = self.lns.to_log(x.clone())
        log_pos = self.lns.to_log(pos_encodings.clone())
        # Add positional encodings to embeddings
        src = self.lns.log_add(log_x, log_pos)
        logging.debug(f"src Shape: {src.shape}") 
        src = self.lns.from_log(src.clone())
        if torch.isnan(src).any() or torch.isinf(src).any():
            logging.error("NaN or Inf detected in model!")
            logging.error(f"src: {src}")
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(device)
        tgt_mask = pad_mask(tgt_mask, seq_len)  # Pad to match the full seq_len
        logging.debug(f"tgt_mask Shape: {tgt_mask.shape}") 
 
        # Forward through transformer encoder (self-attention)
        encode_thought = self.transformer_encoder(src, tgt_mask) #change to Log space
        logging.debug(f"encode_thought Shape: {encode_thought.shape}") 
        if torch.isnan(encode_thought).any() or torch.isinf(encode_thought).any():
            logging.error("NaN or Inf detected in model!")
            logging.error(f"encode_thought: {encode_thought}")
       
        # Prepare for decoding
        tgt_input = torch.zeros_like(input_ids).fill_(self.vocab_size-1)  # <start> token
        # Generate position indices and expand for embedding dimension
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).unsqueeze(-1)
        positions = positions.expand(batch_size, seq_length, embedding_dim)
        tgt_pos_encodings = self.pos_encoder(positions)
        logging.debug(f"tgt_pos_encodings shape: {tgt_pos_encodings.shape}")

        tgt_pos_encodings = tgt_pos_encodings.expand(batch_size, seq_length, -1)
        log_tgt_pos = self.lns.to_log(tgt_pos_encodings.clone())
        # Log space for decoder inputs
        logging.debug(f"tgt_pos_encodings Shape2: {tgt_pos_encodings.shape}") 
        log_tgt_input = self.lns.to_log(self.embedding(tgt_input))
        logging.debug(f"log_tgt_input Shape: {log_tgt_input.shape}") 
        log_tgt_input = self.lns.log_add(log_tgt_input, log_tgt_pos)
        logging.debug(f"log_tgt_input Shape2: {log_tgt_input.shape}") 
        # Generate memory mask as [seq_len, seq_len]
        memory_mask = torch.zeros(seq_length, seq_length, device=device)
        memory_mask = memory_mask.masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask != float('-inf'), 0)
        logging.debug(f"memory_mask Shape: {memory_mask.shape}")

        tgt_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1) ## for autoregressive gpt style decoding add this to decoded_output
        tgt_input = self.lns.from_log(log_tgt_input.clone())
        # Forward through transformer decoder in log space
        decoded_output = self.decoder(tgt_input, encode_thought, tgt_mask=tgt_mask.to(device), memory_mask=memory_mask.to(device))
        logging.debug(f"Decoded Output Shape: {decoded_output.shape}")
        if torch.isnan(decoded_output).any() or torch.isinf(decoded_output).any():
            logging.error("NaN or Inf detected in model!")
            logging.error(f"tgt_emb: {decoded_output}")

        final_output_logits = self.output_projection(decoded_output)
        logging.debug(f"Final Output Logits Shape: {final_output_logits.shape}")


        return final_output_logits

class LayerNormLNS(nn.Module):
    """
    LayerNorm for Logarithmic Number System (LNS).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNormLNS, self).__init__()
        self.eps = eps
        self.lns = LogarithmicNumberSystem()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        logging.debug(f"shape of x for layernorm {x.shape}")

        # Compute mean in log space
        log_mean = log_sum_exp(x, dim=-1, keepdim=True) - math.log(x.size(-1))
        logging.debug(f"shape of log_mean for layernorm {log_mean.shape}")
        
        # Compute variance in log space
        log_var = self.lns.log_sub(
            log_sum_exp(self.lns.log_mul(2 * x, x), dim=-1, keepdim=True) - math.log(x.size(-1)),
            self.lns.log_mul(2 * log_mean, log_mean)
        )
        logging.debug(f"shape of log_var for layernorm {log_var.shape}")

        # Normalize
        log_std = 0.5 * log_var.expand_as(x)  # Ensure log_std has the same shape as x
        normed_x = self.lns.log_div(self.lns.log_sub(x, log_mean.expand_as(x)), log_std)
        logging.debug(f"shape of normed_x for layernorm {normed_x.shape}")

        # Apply gamma and beta
        log_gamma = self.lns.to_log(self.gamma).view(1, 1, -1).expand_as(normed_x)
        log_beta = self.lns.to_log(self.beta).view(1, 1, -1).expand_as(normed_x)
        scaled_x = self.lns.log_mul(normed_x, log_gamma)
        logging.debug(f"shape of scaled_x for layernorm {scaled_x.shape}")

        output = self.lns.log_add(scaled_x, log_beta)
        logging.debug(f"shape of output for layernorm {output.shape}")

        
        return self.lns.from_log(output)

class DropoutLNS(nn.Module):
    """
    Dropout for Logarithmic Number System (LNS).
    """
    def __init__(self, p=0.1):
        super(DropoutLNS, self).__init__()
        self.p = p
        self.lns = LogarithmicNumberSystem()

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        logging.debug(f"shape of x for dropout {x.shape}")

        # Create dropout mask
        mask = (torch.rand_like(x) > self.p).float()
        logging.debug(f"shape of mask for dropout {mask.shape}")

        # Convert mask to log space
        log_mask = self.lns.to_log(mask)
        logging.debug(f"shape of log_mask for dropout {log_mask.shape}")

        # Apply dropout in log space
        log_x = self.lns.to_log(x)
        logging.debug(f"shape of log_x for dropout {log_x.shape}")

        dropped_x = self.lns.log_add(log_x, log_mask)
        logging.debug(f"shape of dropped_x for dropout {dropped_x.shape}")
        
        return self.lns.from_log(dropped_x)

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
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

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
        self.tokenizer_wrapper = TokenizerWrapper(tokenizer, seq_len=seq_length, shift_decoder=False, device=device)
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


def propagate_embedding_size(new_model, new_dim):
    """
    Propagates the new embedding size throughout the model's layers.
    """
    # Update positional encoding if it exists
    if hasattr(new_model, 'pos_encoder'):
        new_model.pos_encoder = PositionalEncoding(new_dim, dropout=0.1, max_len=seq_len)
    
    # Update Transformer layers
    if hasattr(new_model, 'transformer') and isinstance(new_model.transformer, nn.Transformer):
        # Reinitialize the transformer with the new dimension
        new_model.transformer = nn.Transformer(
            d_model=new_dim,
            nhead=new_model.transformer.encoder.layers[0].self_attn.num_heads,
            num_encoder_layers=len(new_model.transformer.encoder.layers),
            num_decoder_layers=len(new_model.transformer.decoder.layers),
            dim_feedforward=new_model.transformer.encoder.layers[0].linear1.out_features,
            dropout=new_model.transformer.encoder.layers[0].dropout.p,
            batch_first=True
        )
    
    # Update output projection layer
    if hasattr(new_model, 'fc_out'):
        new_model.fc_out = nn.Linear(new_dim, new_model.fc_out.out_features)

    # Update all MultiheadAttention layers
    if hasattr(new_model.transformer.encoder, 'layers'):
        for layer in new_model.transformer.encoder.layers:
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, nn.MultiheadAttention):
                layer.self_attn.embed_dim = new_dim
                layer.self_attn.kdim = new_dim
                layer.self_attn.vdim = new_dim
                layer.self_attn.q_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
                layer.self_attn.k_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
                layer.self_attn.v_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
    if hasattr(new_model.transformer.decoder, 'layers'):
        for layer in new_model.transformer.decoder.layers:
            if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, nn.MultiheadAttention):
                layer.self_attn.embed_dim = new_dim
                layer.self_attn.kdim = new_dim
                layer.self_attn.vdim = new_dim
                layer.self_attn.q_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
                layer.self_attn.k_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))
                layer.self_attn.v_proj_weight = nn.Parameter(torch.randn(new_dim, new_dim))


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
        if architecture == "Reasoning Model LNS":

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
            if architecture == "Reasoning Model LNS":

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

                if architecture == "Reasoning Model LNS":

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
        self.learning_rate = tk.DoubleVar(value=0.001)
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
        self.use_genetic_algo = "Genetic Algorithm"  # default to optim
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
        self.architecture = tk.StringVar(value="Reasoning Model")
        ttk.Label(transformer_frame, text="Select Architecture:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.architecture, values=["Reasoning Model", "Reasoning Model LNS"], state="readonly").grid(row=0, column=3)

        ttk.Button(transformer_frame, text="Add Layer", command=self.add_layer).grid(row=4, column=0, pady=5)
        ttk.Button(transformer_frame, text="Save Transformer and Model", command=self.save_transformer_and_model).grid(row=1, column=3, pady=5)
        ttk.Button(transformer_frame, text="Load Transformer", command=self.load_transformer).grid(row=1, column=2, pady=5)
        ttk.Button(transformer_frame, text="Initialize/Load Model", command=self.load_model).grid(row=2, column=3, pady=5)
        self.genetic_algo_var = tk.StringVar(value="GHR Optim")
        ttk.Label(transformer_frame, text="Algo:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(transformer_frame, textvariable=self.genetic_algo_var, values=["GHR Optim", "Genetic Algorithm"], state="readonly").grid(row=0, column=4)

        # Data Selection Frame
        data_frame = ttk.LabelFrame(self.root, text="Data Selection", padding=(10, 10))
        data_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(data_frame, text="Select Dataset Directory", command=self.select_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5)
        ttk.Button(data_frame, text="Load Tokenizer", command=self.load_tokenizer).pack(pady=5)

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
                if self.architecture.get() == "Reasoning Model LNS":
                    max_generated = 50
                    generated_tokens = []
                    generated = []
                    repetition_penalty = 1.2  # Adjust for stronger penalty
                    top_p = 0.9  # Cumulative probability threshold

                    self.model.eval()
                    tokenizer = self.tokenizer
                    with torch.no_grad():
                        # Tokenize prompt → fixed encoder input
                        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                        encoder_input_len = input_ids.size(1)

                        # Pad encoder input to max model length
                        if encoder_input_len < seq_len:
                            pad_len = seq_len - encoder_input_len
                            pad_token_id = tokenizer.pad_token_id or 0
                            padding = torch.full((1, pad_len), pad_token_id, dtype=torch.long).to(device)
                            input_ids = torch.cat([input_ids, padding], dim=1)
                        else:
                            input_ids = input_ids[:, :seq_len]

                        # Encoder is static throughout generation
                        encoder_input_ids = input_ids

                        # Setup initial decoder input
                        bos_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 0
                        tgt_ids = torch.tensor([[bos_token_id]], device=device)

                        for _ in range(max_generated):
                            # Forward pass through model
                            outputs = self.model(encoder_input_ids, tgt_ids)
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
                            print(tgt_ids)
                            # Pad if needed to align with model
                            if tgt_ids.size(1) > seq_len:
                                tgt_ids = tgt_ids[:, -seq_len:]
                            generated.append(self.tokenizer.decode(next_token_id[0].tolist()))


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

    def resize_checkpoint_weights(self, state_dict, new_vocab_size, embed_size):
        """
        Resize checkpoint weights to match the current model's dimensions.
        """
        # This method may need to be updated depending on the model's state_dict keys
        return state_dict

    def select_model_file(self):
        self.model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Model Files", "*.pth;*.json"), ("All files", "*.*")]
        )
        if self.model_path:
            if self.model_path.endswith('.json'):
                # Load model configuration
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                # Update GUI parameters
                self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                self.num_heads.set(config.get("num_heads", self.num_heads.get()))
                self.architecture.set(config.get("architecture", self.architecture.get()))
                messagebox.showinfo("Success", f"Model configuration loaded from: {self.model_path}")
            elif self.model_path.endswith('.pth'):
                # Load model weights
                config_directory = os.path.dirname(self.model_path)
                config_path = os.path.join(config_directory, 'model_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    # Update GUI parameters
                    self.vocab_size.set(config.get("vocab_size", self.vocab_size.get()))
                    self.hidden_size.set(config.get("embed_size", self.hidden_size.get()))
                    self.num_layers.set(config.get("num_layers", self.num_layers.get()))
                    self.num_heads.set(config.get("num_heads", self.num_heads.get()))

                    self.architecture.set(config.get("architecture", self.architecture.get()))
                    # Load the model
                    self.load_model()
                    # Load model state
                    state_dict = torch.load(self.model_path, map_location=self.device)

                    self.model.load_state_dict(state_dict, strict=False)
                    messagebox.showinfo("Success", f"Model weights and configuration loaded from: {self.model_path}")
                else:
                    messagebox.showwarning("Warning", "Model configuration file not found. Please ensure the configuration is set correctly.")
            else:
                messagebox.showerror("Error", "Unsupported file format selected.")

    def save_transformer_and_model(self):
        if not self.model:
            messagebox.showerror("Error", "Model has not been initialized. Please initialize the model first.")
            return
        if not self.tokenizer:
            messagebox.showerror("Error", "Tokenizer has not been initialized. Please load a tokenizer first.")
            return

        transformer_data = {
            "vocab_size": self.vocab_size.get(),
            "embed_size": self.hidden_size.get(),
            "hidden_size": self.hidden_size.get(),
            "num_layers": self.num_layers.get(),
            "architecture": self.architecture.get(),
            "num_heads": self.num_heads.get(),
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory:
            # Save configuration
            config_path = os.path.join(directory, "model_config.json")
            with open(config_path, "w") as file:
                json.dump(transformer_data, file, indent=4)

            # Save weights
            if self.architecture.get() == "Reasoning Model":
                model_file_name = 'reasoning_model.pth'
            elif self.architecture.get() == "Reasoning Model LNS":
                model_file_name = 'reasoning_model_lns.pth'
            else:
                messagebox.showerror("Error", f"Unsupported architecture: {self.architecture.get()}")
                return

            model_path = os.path.join(directory, model_file_name)
            torch.save(self.model.state_dict(), model_path)

            # Save tokenizer
            self.tokenizer.save_pretrained(directory)

            messagebox.showinfo("Success", "Model, tokenizer, and configuration saved successfully!")
            logging.info("Model, tokenizer, and configuration saved successfully.")

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
        self.tokenizer_path = filedialog.askopenfilename(
            title="Select Tokenizer File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if self.tokenizer_path:
            messagebox.showinfo("Success", f"Tokenizer file selected: {self.tokenizer_path}")

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


    def load_tokenizer(self):
        try:
            self.tokenizer_path = filedialog.askopenfilename(
                title="Select Tokenizer File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not self.tokenizer_path or not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError("Tokenizer file not selected or does not exist.")

            # Load the PreTrainedTokenizerFast from file.
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file = self.tokenizer_path)
            logging.info(f"Tokenizer loaded from {self.tokenizer_path}")

            # If a special tokens map exists, load and add them.
            special_tokens_path = os.path.join(os.path.dirname(self.tokenizer_path), "special_tokens_map.json")
            if os.path.exists(special_tokens_path):
                with open(special_tokens_path, "r", encoding="utf-8") as file:
                    special_tokens = json.load(file)
                # Convert nested dicts to AddedToken if needed.
                for key, value in special_tokens.items():
                    if isinstance(value, dict):
                        special_tokens[key] = AddedToken(value["content"],
                                                        lstrip=value.get("lstrip", False),
                                                        rstrip=value.get("rstrip", False))
                    elif not isinstance(value, (str, AddedToken)):
                        raise ValueError(f"Invalid token format for key {key}: {value}")
                self.tokenizer.add_special_tokens(special_tokens)
                logging.info(f"Special tokens added: {special_tokens}")

            # Load tokenizer configuration if available.
            tokenizer_config_path = os.path.join(os.path.dirname(self.tokenizer_path), "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, "r", encoding="utf-8") as file:
                    tokenizer_config = json.load(file)
                    self.tokenizer.init_kwargs.update(tokenizer_config)
                    if "model_max_length" in tokenizer_config:
                        self.tokenizer.model_max_length = tokenizer_config["model_max_length"]
                    logging.info(f"Tokenizer configuration loaded: {tokenizer_config}")

            # Ensure a reasonable model_max_length is set.
            if not hasattr(self.tokenizer, "model_max_length") or self.tokenizer.model_max_length > 1024 * 1024:
                self.tokenizer.model_max_length = seq_len  # Default value; ensure seq_len is defined
            logging.info(f"Model max length set to: {self.tokenizer.model_max_length}")

            # Log the vocabulary size.
            tokenizer_vocab_size = len(self.tokenizer)
            logging.info(f"Loaded tokenizer vocabulary size: {tokenizer_vocab_size}")
            self.vocab_size.set(tokenizer_vocab_size)

            # Ensure special tokens are correctly set.
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = "<PAD>"
                self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<PAD>")
                logging.warning("Pad token was not set. Defaulting to <PAD>.")
            if not self.tokenizer.unk_token:
                self.tokenizer.unk_token = "<UNK>"
                self.tokenizer.unk_token_id = self.tokenizer.convert_tokens_to_ids("<UNK>")
                logging.warning("UNK token was not set. Defaulting to <UNK>.")
            if not self.tokenizer.bos_token:
                self.tokenizer.bos_token = "<BOS>"
                self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<BOS>")
                logging.warning("BOS token was not set. Defaulting to <BOS>.")
            if not self.tokenizer.eos_token:
                self.tokenizer.eos_token = "<EOS>"
                self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<EOS>")
                logging.warning("EOS token was not set. Defaulting to <EOS>.")

            print("Special tokens map:", self.tokenizer.special_tokens_map)
            print("Pad token ID:", self.tokenizer.pad_token_id)
            print("Model max length:", self.tokenizer.model_max_length)

        except Exception as e:
            logging.error(f"Failed to load tokenizer: {str(e)}")
            messagebox.showerror("Error", f"Failed to load tokenizer: {str(e)}")


    def add_layer(self):
        layer_type = simpledialog.askstring("Layer Type", "Enter layer type (e.g., attention, feed_forward)")
        if layer_type:
            layer_config = {
                "type": layer_type,
                "parameters": {}  # Placeholder for future parameter configuration
            }
            self.layers.append(layer_config)
            messagebox.showinfo("Layer Added", f"Layer of type '{layer_type}' added.")

    def save_transformer(self):
        transformer_data = {
            "num_parameters": self.num_parameters.get(),
            "layers": self.layers
        }

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(transformer_data, file, indent=4)
            messagebox.showinfo("Save", "Transformer saved successfully!")
            logging.info(f"Number of layers in the model: {len(self.model.transformer_encoder.layers)}")

    def load_transformer(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                transformer_data = json.load(file)
            self.num_parameters.set(transformer_data["num_parameters"])
            self.num_heads.set(transformer_data["num_heads"])
            self.layers = transformer_data["layers"]
            messagebox.showinfo("Success", "Transformer loaded successfully")

    def load_model(self):
        try:
            if not self.tokenizer:
                vocab_size = self.vocab_size.get()
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

            elif self.architecture.get() == "Reasoning Model LNS":
                self.model = Transformer_Model_LNS(
                    vocab_size=vocab_size,
                    embedding_dim=self.hidden_size.get(),
                    hidden_dim=self.hidden_size.get(),
                    num_layers=self.num_layers.get(),
                    num_heads=self.num_heads.get(),
                    seq_length=seq_len
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
                self.model.load_state_dict(state_dict, strict=False)
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


            logging.info("dataset torched")
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=int(self.batch_size.get()),
                    shuffle=True,
                    num_workers=0,    # Uses multiple CPU threads for data loading
                    pin_memory=False,
                    collate_fn=collate_fn
                )
            logging.info("dataloader defined")
            ##chunked vs. standard else complete

            # Adjust learning rate based on architecture
            total_params = self.num_parameters.get()
            lr = self.learning_rate.get()
            logging.info(f"Learning Rate: {lr} for total parameters: {total_params}")

            # Learning rate scheduler
            total_steps = self.epochs.get() * len(dataloader)
            logging.info(f"Total training steps: {total_steps}")
            # Separate parameters based on their shape.

            # Create two optimizers:
            #Enable for standard optimizer/scheduler
            #num_warmup_steps = total_steps // 10  # Warmup for 10% of training
            #scheduler = self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

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
                for batch_idx, (full_tokens) in enumerate(dataloader):  # [batch, padded_len]

                    if self.stop_training.is_set():
                            logging.info("Training stopped by user.")
                            messagebox.showinfo("Info", "Training stopped by user.")
                            return
                    logging.debug(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}')

                        # Move batches and targets to the correct device 
                    full_tokens = full_tokens.to(device, non_blocking=True)

                    max_len = full_tokens.shape[1]
                        # Logging epoch and batch info
                    logging.debug(f'Batch shape: {max_len}')  # (batch_size, seq_len)
                    logging.debug(f'Using device: {self.device}')


                    optimizer.zero_grad()
                    batch_loss = 0
                    step_count = 0

                    architecture = self.architecture.get()
                    for t in range(2, max_len):
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

                        del targets_flat
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
                                            
                                def forward_fn(tgt):
                                    return self.model(src, tgt, mode="train")
                                    # Forward pass
                                try:
                                        if architecture == "Reasoning Model LNS":

                                            logits = self.model(src, tgt_ids)

                                        else:
                                            logits = cp.checkpoint(forward_fn, tgt_ids, use_reentrant=False)
                                except Exception as e:
                                            raise ValueError(f"forward pass failed for {str(e)}")

                                # Reshape to [batch * seq_len, vocab] and filter by mask
                                logits_flat = logits.reshape(-1, logits.shape[-1])                    # [B*T, V]
                                logits_filtered = logits_flat[active_mask_flat]                      # [N, V]

                                if logits_filtered.size(0) == 0:
                                    continue  # skip if nothing to train on this step


                                logging.debug(f"Shape of logits: {logits.shape}")
                                    # Assume batch_labels are tensors of shape [batch_size, seq_len, vocab_size]
                                logging.debug(f"logits reshaped Shape: {logits_filtered.shape}")

                                step_loss = loss_fn(logits_filtered, targets_filtered)

                                step_loss.backward()
                                logging.info(f"Loss computed: {step_loss.item()}")
                                # ✅ Check for NaNs in loss
                                if torch.isnan(step_loss).any():
                                    logging.warning("⚠️ Skipping optimizer step due to NaN loss.")
                                # Backward pass and optimization

                                logging.info("Loss backward computed")
                                    
                                print(f"Iteration {t}, Loss: {step_loss.item()}")
                                if torch.isnan(step_loss) or torch.isinf(step_loss):
                                    print("🚨 Warning: NaN or Inf detected in loss! Skipping update.")
                                    return
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
                    print(f"Batch iteration {n}, Loss: {avg_loss}")
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

        save_directory = filedialog.askdirectory(title="Select Save Directory")
        if save_directory:
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
            elif self.architecture.get() == "Reasoning Model LNS":
                model_file_name = 'reasoning_model_lns.pth'
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
