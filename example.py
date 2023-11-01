import torch
import distquant
torch.manual_seed(0)

# Generate 100k points of data following normal + 0.2 * uniform
data = torch.randn(100_000)
data = data + 0.2 * torch.rand_like(data)

q_data, info = distquant.quantize(
    data,
    k=7,              # 7 bit quantization
    dist=None,        # Automatically search for the optimal distributional assumption
    packbits=True,    # Packs the results into int64. For `k=7`, we have 9 values per int64
)

print(info)

print('Original footprint:  ', 4 * data.shape[0], 'bytes')    # Original footprint:   400000 bytes
print('Quantized footprint: ', 8 * q_data.shape[0], 'bytes')  # Quantized footprint:  88896 bytes

deq_data = distquant.dequantize(q_data, **info)
