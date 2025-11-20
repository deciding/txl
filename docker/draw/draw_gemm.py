import numpy as np
import matplotlib.pyplot as plt

gemm_k = np.array([256, 512, 1024, 2048, 4096, 8192, 16384])

# ------- 示例数据，自行替换 -------
cublas_fp16 = np.array([517, 626, 712, 717, 697, 680, 667])
# tawa_fp16   = np.array([580, 760, 780, 770, 780, 760, 740])
txl_fp16 = np.array([473, 615, 705, 739, 748, 701, 694])
triton_fp16 = np.array([470, 603, 680, 678, 670, 640, 630])
tile_fp16   = np.array([300, 420, 600, 690, 700, 720, 740])
tk_fp16     = np.array([400, 680, 680, 709, 780, 788, 798])

cublas_fp8 = np.array([1300, 1400, 1500, 1550, 1500, 1500, 1400])
tawa_fp8   = np.array([900,  1470, 1600, 1600, 1550, 1500, 1400])
# fp8 运行不了
txl_fp8 = np.array([900,  1470, 1600, 1600, 1550, 1500, 1400])
triton_fp8 = np.array([600,  1000, 1450, 1500, 1500, 1500, 1400])
tile_fp8   = np.array([700,  1400, 1500, 1600, 1550, 1500, 1400])
tk_fp8     = np.array([600,  1200, 1600, 1500, 1500, 1450, 1400])
# -------------------------------

theoretical_fp16 = 1000
theoretical_fp8  = 2000

bar_width = 0.14
x = np.arange(len(gemm_k))

fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=False)

# ------------ 左图 FP16 ------------
ax = axes[0]
ax.axhline(theoretical_fp16, color='gray', linewidth=3, label='Theoretical Peak')

ax.bar(x - 2*bar_width, cublas_fp16, bar_width,
       label='cuBLAS', color='#f1c40f', hatch='//')
ax.bar(x - 1*bar_width, txl_fp16,   bar_width,
       label='Txl', color='#e74c3c')
ax.bar(x + 0*bar_width, triton_fp16, bar_width,
       label='Triton', color='#1abc9c')
ax.bar(x + 1*bar_width, tile_fp16,   bar_width,
       label='TileLang', color='#ff6fb3')
ax.bar(x + 2*bar_width, tk_fp16,     bar_width,
       label='ThunderKittens', color='#3498db')

ax.set_title('FP16')
ax.set_ylabel('Throughput (TFLOPs/s)')
ax.set_xticks(x)
ax.set_xticklabels(gemm_k)
ax.set_xlabel('GEMM K size')
ax.grid(axis='y', linestyle='--', alpha=0.4)

left_ylim = 1200
ax.set_ylim(0, left_ylim)
ax.set_yticks([0, 200, 400, 600, 800, 1000, 1200])

# ------------ 右图 FP8 ------------
ax = axes[1]
ax.axhline(theoretical_fp8, color='gray', linewidth=3)

ax.bar(x - 2*bar_width, cublas_fp8, bar_width,
       color='#f1c40f', hatch='//')
ax.bar(x - 1*bar_width, txl_fp8,   bar_width,
       color='#e74c3c')
ax.bar(x + 0*bar_width, triton_fp8, bar_width,
       color='#1abc9c')
ax.bar(x + 1*bar_width, tile_fp8,   bar_width,
       color='#ff6fb3')
ax.bar(x + 2*bar_width, tk_fp8,     bar_width,
       color='#3498db')

ax.set_title('FP8')
ax.set_xticks(x)
ax.set_xticklabels(gemm_k)
ax.set_xlabel('GEMM K size')
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 关键：让 2000 所在位置与 1000 对齐
right_ylim = theoretical_fp8 * left_ylim / theoretical_fp16  # = 2400
ax.set_ylim(0, right_ylim)
ax.set_yticks(np.arange(0, 2001, 500))   # 0, 500, 1000, 1500, 2000

# ------------ 顶部 legend & 布局 ------------
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=6,
           bbox_to_anchor=(0.5, 0.98))

# plt.subplots_adjust(top=0.78, bottom=0.18, left=0.07, right=0.98, wspace=0.25)
plt.subplots_adjust(top=0.78, bottom=0.18, left=0.08, right=0.98,
                    wspace=0.15, hspace=0.35)

# plt.tight_layout()
plt.show()

# m=8192, n=8192