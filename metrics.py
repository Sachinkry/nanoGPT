import time
import torch
import psutil
import pynvml
import os

class MetricsTracker:
    def __init__(self, model, batch_size, block_size, eval_iters, filename='metrics_logs.txt'):
        self.model = model
        self.batch_size = batch_size
        self.block_size = block_size
        self.eval_iters = eval_iters
        self.filename = filename
        self.start_time = time.time()
        
        # Initialize NVIDIA Management Library (if GPU is available)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            pynvml.nvmlInit()
            self.device_id = torch.cuda.current_device()
        
    def tokens_per_second(self, num_tokens):
        """Calculate tokens per second during training."""
        elapsed_time = time.time() - self.start_time
        return num_tokens / elapsed_time
    
    def time_per_token(self, num_tokens):
        """Calculate time per token during training."""
        elapsed_time = time.time() - self.start_time
        time_per_token = (elapsed_time*1000) / num_tokens if num_tokens > 0 else float('inf')
        return time_per_token, elapsed_time

    def gpu_utilization(self):
        """Get the current GPU utilization."""
        if self.use_gpu:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        return 0

    def memory_usage(self):
        """Get memory usage by the current process."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 3)  # Convert to GB
    
    def calculate_memory_usage(self):
        """Calculate the memory required to store the model parameters in MB."""
        num_params = sum(p.numel() for p in self.model.parameters())
        # Assuming parameters are stored as float32 (4 bytes)
        memory_usage_mb = (num_params * 4) / (1024 ** 2)  # Convert bytes to MB
        return memory_usage_mb
    
    def log_metrics(self, train_loss, val_loss, num_tokens):
        """Log training metrics to a file."""
        tokens_per_sec = self.tokens_per_second(num_tokens)
        time_per_token, elapsed_time = self.time_per_token(num_tokens)
        gpu_util = self.gpu_utilization()
        memory = self.memory_usage()
        # with open(self.filename, 'a') as f:
        #     f.write(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
        #     f.write(f"Tokens/Second: {tokens_per_sec:.2f}, GPU Utilization: {gpu_util}%, Memory Usage: {memory:.2f} GB\n\n")
        print(f"[Metrics]:")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Total tokens: {num_tokens}")
        print(f"Tokens/Second: {tokens_per_sec:.2f}, Time/Token: {time_per_token:.6f} ms")
        print(f"Training Time: {(elapsed_time/60):.2f} mins")
        print(f"GPU Utilization: {gpu_util}%, Memory Usage: {memory:.2f} GB")
    
    def compare_with_baseline(self, baseline_filename='baseline_logs.txt'):
        """Compare current metrics with baseline metrics."""
        if not os.path.exists(baseline_filename):
            print("Baseline file not found.")
            return
        
        with open(self.filename, 'r') as f:
            current_log = f.readlines()
        with open(baseline_filename, 'r') as f:
            baseline_log = f.readlines()
        
        print("\n--- Comparison with Baseline ---")
        print("Current Run:")
        print("".join(current_log[-5:]))
        print("Baseline:")
        print("".join(baseline_log[-5:]))
