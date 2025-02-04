import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# 打印设备信息
device = cuda.Device(0)  # 获取第一个 GPU
print(f"Device name: {device.name()}")
print(f"Compute capability: {device.compute_capability()}")

# 测试简单 CUDA kernel
mod = SourceModule("""
__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
""")

print("CUDA Kernel compiled successfully!")
