import numpy as np
import matplotlib.pyplot as plt

# 横轴：上下四张图的 context length
ctx = np.array([1024, 2048, 4096, 8192, 16384, 32768])
x = np.arange(len(ctx))  # 真正用于画图的位置：0,1,2,...

# ====================== 示例数据，按需替换 ======================
# FP16, causal = False, sq=1
fp16_nc_fm   = np.array([436, 510, 543, 564, 582, 601])
fp16_nc_txl  = np.array([401, 490, 518, 535, 538, 561])
fp16_nc_triton= np.array([19, 28, 34, 38, 44, 46])
fp16_nc_tile  = np.array([237, 412, 459, 473, 498, 477])
fp16_nc_fi    = np.array([406, 491, 527, 532, 528, 552])
# FP16, causal = False, sq=2
fp16_nc2_fm   = np.array([521, 591, 621, 628, 579, 626])
fp16_nc2_txl  = np.array([475, 531, 554, 557, 565, 587])
fp16_nc2_fi    = np.array([486, 534, 535, 545, 536, 539])
# ===============================================================

methods_q1 = ["FlashMLA", "Txl", "Triton", "TileLang", "Flashinfer"]
methods_q2 = ["FlashMLA", "Txl", "Flashinfer"]

colors = {
    "FlashMLA": "#f1c40f",
    "Txl":           "#e74c3c",
    "Triton":         "#1abc9c",
    "TileLang":       "#ff6fb3",
    "Flashinfer": "#e67e22",
}

data = {
    ("FP16, causal=False, s_q=1"): {
        "FlashMLA": fp16_nc_fm,
        "Txl":           fp16_nc_txl,
        "Triton":         fp16_nc_triton,
        "TileLang":       fp16_nc_tile,
        "Flashinfer": fp16_nc_fi,
    },
    ("FP16, causal=False, s_q=2"): {
        "FlashMLA": fp16_nc2_fm,
        "Txl":           fp16_nc2_txl,
        "Flashinfer": fp16_nc2_fi,
    },
}

# fig, axes = plt.subplots(2, 2, figsize=(12, 4), sharex=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

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
    ax[0],
    "FP16, causal=false, s_q=1",
    data[("FP16, causal=False, s_q=1")],
    ylim=(0, 800),
    yticks=[0, 200, 400, 600, 800],
    methods=methods_q1,
)

plot_panel(
    ax[1],
    "FP16, causal=false, s_q=2",
    data[("FP16, causal=False, s_q=2")],
    ylim=(0, 800),
    yticks=[0, 200, 400, 600, 800],
    methods=methods_q2,
)
# axes[0, 0].set_ylabel("Throughput (TFLOPs/s)")
# axes[1, 0].set_ylabel("Throughput (TFLOPs/s)")
# axes[1, 0].set_xlabel("Context length")
# axes[1, 1].set_xlabel("Context length")

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 0.98),
)

ax[0].set_ylabel("Throughput (TFLOPs/s)")
ax[0].set_xlabel("Context length")
ax[1].set_xlabel("Context length")

plt.subplots_adjust(top=0.82)

plt.show()

# b=132, s_q=2, h_q=128, h_kv=1, d=576, dv=512, causal=False, dtype=torch.float16
