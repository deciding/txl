import matplotlib.pyplot as plt

# 原始数据
length_labels = [256, 512, 1024, 2048, 4096]          # 用来当 x 轴刻度标签
x = list(range(len(length_labels)))       # 实际用于绘图的 x 坐标：0,1,2

torch_times  = [14.92, 28.43, 55.59]
triton_times = [5.93, 13.09, 32.74, 76.94, 202.10]
txl_times    = [8.28, 10.53, 22.13, 45.38, 102.71]

plt.figure(figsize=(6, 4))

# 用 0,1,2 作为横坐标，这样间隔一定相同
plt.plot(list(range(len(torch_times))), torch_times,  marker='o', label='PyTorch')
plt.plot(list(range(len(triton_times))), triton_times, marker='s', label='Triton')
plt.plot(list(range(len(txl_times))), txl_times,    marker='^', label='Txl')

plt.xlabel('Context Length')
plt.ylabel('Latency (ms)')
plt.title('GPT2-Style End-to-End Inference Latency Comparison')

# 把 0,1,2 映射成 256,512,1024
plt.xticks(x, length_labels)

plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

