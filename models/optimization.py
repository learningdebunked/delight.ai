"""
Performance optimization techniques for the SEDS system.
"""
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import numpy as np
import torch
import torch.nn as nn
import torch.jit
from torch.utils.cpp_extension import load
import psutil
import gc
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimizes models for inference and training performance."""
    
    @staticmethod
    def optimize_model(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        use_amp: bool = True,
        use_torchscript: bool = True,
        use_cuda_graph: bool = False,
        jit_compile: bool = True
    ) -> nn.Module:
        """
        Apply various optimizations to a PyTorch model.
        
        Args:
            model: The PyTorch model to optimize
            input_shape: Expected input shape (batch_size, ...)
            use_amp: Whether to use automatic mixed precision
            use_torchscript: Whether to convert to TorchScript
            use_cuda_graph: Whether to use CUDA graphs (CUDA only)
            jit_compile: Whether to use PyTorch's JIT compilation
            
        Returns:
            Optimized model
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Apply optimizations
        if use_amp:
            model = ModelOptimizer._apply_amp(model)
            
        if jit_compile and hasattr(torch, 'compile') and torch.cuda.is_available():
            model = torch.compile(model, mode='max-autotune')
        
        if use_torchscript:
            try:
                with torch.no_grad():
                    # Create dummy input with the right shape and device
                    dummy_input = torch.randn(*input_shape, device=device)
                    model = torch.jit.trace(model, dummy_input, check_trace=False)
                    model = torch.jit.freeze(model)
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
        
        if use_cuda_graph and torch.cuda.is_available():
            model = ModelOptimizer._apply_cuda_graph(model, input_shape)
        
        return model
    
    @staticmethod
    def _apply_amp(model: nn.Module) -> nn.Module:
        """Apply Automatic Mixed Precision (AMP) to the model."""
        if not hasattr(torch.cuda, 'amp') or not torch.cuda.is_available():
            return model
            
        from torch.cuda.amp import autocast
        
        class AMPWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, *args, **kwargs):
                with autocast():
                    return self.model(*args, **kwargs)
        
        return AMPWrapper(model)
    
    @staticmethod
    def _apply_cuda_graph(model: nn.Module, input_shape: Tuple[int, ...]) -> nn.Module:
        """Apply CUDA graph optimization to the model."""
        if not torch.cuda.is_available():
            return model
            
        device = next(model.parameters()).device
        
        # Warmup
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                dummy_input = torch.randn(*input_shape, device=device)
                model(dummy_input)
        torch.cuda.current_stream().wait_stream(s)
        
        # Create graph
        g = torch.cuda.CUDAGraph()
        dummy_input = torch.randn(*input_shape, device=device)
        
        with torch.cuda.graph(g):
            model(dummy_input)
        
        # Wrap model to use the graph
        class CUDAGraphWrapper(nn.Module):
            def __init__(self, model, graph, input_shape):
                super().__init__()
                self.model = model
                self.graph = graph
                self.input_shape = input_shape
                self.static_input = torch.zeros(*input_shape, device=device)
                
            def forward(self, x):
                if x.shape == self.input_shape:
                    self.static_input.copy_(x)
                    self.graph.replay()
                    return self.static_input
                else:
                    return self.model(x)
        
        return CUDAGraphWrapper(model, g, input_shape)
    
    @staticmethod
    def optimize_memory_usage(model: nn.Module) -> None:
        """Optimize model memory usage."""
        # Empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Set all parameters to not require gradients if not training
        for param in model.parameters():
            param.requires_grad = False
            
        # Run garbage collector
        gc.collect()
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time and memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time execution
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        logger.info(
            f"Function {func.__name__} executed in {elapsed_time:.4f}s, "
            f"memory used: {mem_used:.2f}MB"
        )
        
        return result
    return wrapper


class GradientAccumulator:
    """Accumulates gradients over multiple batches before performing an update."""
    
    def __init__(self, model: nn.Module, accumulation_steps: int = 4):
        """
        Initialize the gradient accumulator.
        
        Args:
            model: The model whose gradients will be accumulated
            accumulation_steps: Number of steps to accumulate gradients over
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def zero_grad(self):
        """Reset accumulated gradients."""
        self.current_step = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> bool:
        """
        Accumulate gradients and perform an optimization step if needed.
        
        Args:
            loss: The loss tensor
            optimizer: The optimizer to use for the step
            
        Returns:
            bool: True if a step was performed, False otherwise
        """
        # Scale loss for accumulation
        (loss / self.accumulation_steps).backward()
        self.current_step += 1
        
        # Perform optimization step if we've accumulated enough gradients
        if self.current_step >= self.accumulation_steps:
            optimizer.step()
            self.zero_grad()
            return True
            
        return False


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient self-attention implementation."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Efficient attention with optional memory optimizations.
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.size(1)
        
        # Project queries, keys and values
        q = self.q_proj(query).view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Apply attention dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        if need_weights:
            return attn_output, attn_weights
        return attn_output, None
