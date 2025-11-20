import numpy as np
import matplotlib.pyplot as plt

# 横轴：上下四张图的 context length
ctx = np.array([1024, 2048, 4096, 8192, 16384, 32768])
x = np.arange(len(ctx))  # 真正用于画图的位置：0,1,2,...

# ====================== 示例数据，按需替换 ======================
# FP16, causal = False
fp16_nc_fm   = np.array([436, 510, 543, 560, 576, 579])
fp16_nc_txl  = np.array([393, 476, 518, 523, 525, 553])
fp16_nc_triton= np.array([20, 78, 98, 112, 136, 150])
fp16_nc_tile  = np.array([237, 430, 459, 465, 465, 463])
fp16_nc_fi    = np.array([290, 320, 360, 387, 350, 362])
# ===============================================================

methods = ["FlashMLA", "Txl", "Triton", "TileLang", "Flashinfer"]

colors = {
    "FlashMLA": "#f1c40f",
    "Txl":           "#e74c3c",
    "Triton":         "#1abc9c",
    "TileLang":       "#ff6fb3",
    "Flashinfer": "#e67e22",
}

data = {
    ("FP16, causal=False"): {
        "FlashMLA": fp16_nc_fm,
        "Txl":           fp16_nc_txl,
        "Triton":         fp16_nc_triton,
        "TileLang":       fp16_nc_tile,
        "Flashinfer": fp16_nc_fi,
    },
}

# fig, axes = plt.subplots(2, 2, figsize=(12, 4), sharex=True)
fig, ax = plt.subplots(figsize=(6, 4))

bar_width = 0.16

def plot_panel(ax, title, panel_data, ylim, yticks, methods):
    count = len(methods)
    offsets = (np.arange(count) - (count - 1) / 2) * bar_width
    for i, m in enumerate(methods):
        vals = panel_data[m]
        if m == "FlashMLA":
            ax.bar(
                x + offsets[i],   # 注意这里用 x，而不是 ctx
                vals,
                bar_width,
                label=m,
                color=colors[m],
                hatch="//",
            )
        else:
            ax.bar(
                x + offsets[i],
                vals,
                bar_width,
                label=m,
                color=colors[m],
            )
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ctx)  # 只把刻度标签写成 1024, 2048...

# 上排：FP16
plot_panel(
    ax,
    "FP16, causal=false",
    data[("FP16, causal=False")],
    ylim=(0, 800),
    yticks=[0, 200, 400, 600, 800],
    methods=methods,
)

# axes[0, 0].set_ylabel("Throughput (TFLOPs/s)")
# axes[1, 0].set_ylabel("Throughput (TFLOPs/s)")
# axes[1, 0].set_xlabel("Context length")
# axes[1, 1].set_xlabel("Context length")

handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 0.98),
)

ax.set_ylabel("Throughput (TFLOPs/s)")
ax.set_xlabel("Context length")

plt.subplots_adjust(top=0.82)

plt.show()

# b=132, s_q=2, h_q=128, h_kv=1, d=576, dv=512, causal=False, dtype=torch.float16
