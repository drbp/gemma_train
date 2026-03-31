import torch
import time

def run_benchmark(size=4096, iterations=100):
    if not torch.cuda.is_available():
        print("Error: CUDA not found. Make sure your 2080 Ti drivers are installed.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    
    # 1. Setup: Use float16 as is standard for LLM training
    # 4096 is a typical 'hidden size' for models like Llama
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    
    # 2. Warmup: Ensures the GPU is at full clock speed
    print(f"Benchmarking: {gpu_name}...")
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # 3. Execution Loop
    start_time = time.time()
    for _ in range(iterations):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 4. Calculations
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    
    # Formula for TFLOPS: (2 * M * N * K * iterations) / (time * 10^12)
    # For a square matrix, it's 2 * size^3
    tflops = (2 * size**3 * iterations) / (total_time * 1e12)
    
    # 5. Memory Check
    max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    
    print("-" * 30)
    print(f"Avg Time per Step: {avg_time_ms:.2f} ms")
    print(f"Real-world Performance: {tflops:.2f} TFLOPS")
    print(f"Peak VRAM used: {max_mem:.2f} MB")
    print("-" * 30)

if __name__ == "__main__":
    run_benchmark()
