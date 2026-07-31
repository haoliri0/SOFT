import symft

print(symft.__version__)
print(symft.simd_backend())
print(symft.cuda_enabled())
print(symft.active_cuda_backend())
