import numpy as np
import matplotlib.pyplot as plt

# 横轴：上下四张图的 context length
ctx = np.array([1024, 2048, 4096, 8192, 16384])
x = np.arange(len(ctx))  # 真正用于画图的位置：0,1,2,...

# ====================== 示例数据，按需替换 ======================
# FP16, causal = False
fp16_nc_fa3   = np.array([570, 600, 610, 630, 640])
fp16_nc_txl  = np.array([484, 544, 578, 597, 608])
fp16_nc_triton= np.array([390, 460, 500, 520, 540])
fp16_nc_tile  = np.array([447, 590, 610, 570, 600])
fp16_nc_tk    = np.array([453, 597, 599, 610, 590])

# FP16, causal = True Txl 还没有支持 causal attention
fp16_c_fa3    = np.array([420, 520, 620, 680, 650])
fp16_c_tawa   = np.array([380, 480, 580, 630, 620])
fp16_c_triton = np.array([320, 420, 500, 550, 540])
fp16_c_tile   = np.array([340, 440, 520, 570, 560])
fp16_c_tk     = np.array([330, 430, 510, 560, 550])

# FP8, causal = False Txl 还没有支持 FP8
fp8_nc_fa3    = np.array([587, 770, 850, 900, 980])
fp8_nc_txl   = np.array([547, 627, 676, 703, 706])
fp8_nc_triton = np.array([520, 610, 690, 716, 723])

# FP8, causal = True Txl 还没有支持 FP8
fp8_c_fa3     = np.array([600, 800, 900, 950, 930])
fp8_c_tawa    = np.array([540, 720, 800, 840, 820])
fp8_c_triton  = np.array([450, 630, 720, 760, 740])
fp8_c_tile    = np.array([470, 650, 740, 780, 760])
fp8_c_tk      = np.array([460, 640, 730, 770, 750])
# ===============================================================

methods_up = ["FA3 (CUTLASS)", "Txl", "Triton", "TileLang", "ThunderKittens"]
methods_down = ["FA3 (CUTLASS)", "Txl", "Triton"]
colors = {
    "FA3 (CUTLASS)": "#f1c40f",
    "Txl":           "#e74c3c",
    "Triton":         "#1abc9c",
    "TileLang":       "#ff6fb3",
    "ThunderKittens": "#3498db",
}

data = {
    ("FP16, causal=False"): {
        "FA3 (CUTLASS)": fp16_nc_fa3,
        "Txl":           fp16_nc_txl,
        "Triton":         fp16_nc_triton,
        "TileLang":       fp16_nc_tile,
        "ThunderKittens": fp16_nc_tk,
    },
    ("FP16, causal=True"): {
        "FA3 (CUTLASS)": fp16_c_fa3,
        "Txl":           fp16_c_tawa,
        "Triton":         fp16_c_triton,
        "TileLang":       fp16_c_tile,
        "ThunderKittens": fp16_c_tk,
    },
    ("FP8, causal=False"): {
        "FA3 (CUTLASS)": fp8_nc_fa3,
        "Txl":           fp8_nc_txl,
        "Triton":         fp8_nc_triton,
    },
    ("FP8, causal=True"): {
        "FA3 (CUTLASS)": fp8_c_fa3,
        "Txl":           fp8_c_tawa,
        "Triton":         fp8_c_triton,
    },
}

fig, axes = plt.subplots(2, 2, figsize=(12, 4), sharex=True)

bar_width = 0.16

def plot_panel(ax, title, panel_data, ylim, yticks, methods):
    count = len(methods)
    offsets = (np.arange(count) - (count - 1) / 2) * bar_width
    for i, m in enumerate(methods):
        vals = panel_data[m]
        if m == "FA3 (CUTLASS)":
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
    axes[0, 0],
    "FP16, causal=false",
    data[("FP16, causal=False")],
    ylim=(0, 800),
    yticks=[0, 200, 400, 600, 800],
    methods=methods_up,
)
plot_panel(
    axes[0, 1],
    "FP16, causal=true",
    data[("FP16, causal=True")],
    ylim=(0, 800),
    yticks=[0, 200, 400, 600, 800],
    methods=methods_up,
)

# 下排：FP8
plot_panel(
    axes[1, 0],
    "FP8, causal=false",
    data[("FP8, causal=False")],
    ylim=(0, 1250),
    yticks=[0, 250, 500, 750, 1000],
    methods=methods_down,
)
plot_panel(
    axes[1, 1],
    "FP8, causal=true",
    data[("FP8, causal=True")],
    ylim=(0, 1250),
    yticks=[0, 250, 500, 750, 1000],
    methods=methods_down,
)

# axes[0, 0].set_ylabel("Throughput (TFLOPs/s)")
# axes[1, 0].set_ylabel("Throughput (TFLOPs/s)")
# axes[1, 0].set_xlabel("Context length")
# axes[1, 1].set_xlabel("Context length")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.5, 0.98),
)

# 全局坐标轴标签，只写一次，自动在整张图居中对齐
fig.supylabel("Throughput (TFLOPs/s)")     # 左侧垂直居中
fig.supxlabel("Context length")           # 底部水平居中

plt.subplots_adjust(top=0.82, bottom=0.12, left=0.08, right=0.98,
                    wspace=0.15, hspace=0.35)

# plt.tight_layout()
plt.show()

# batch4-head32-d128