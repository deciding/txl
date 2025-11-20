import numpy as np
import matplotlib.pyplot as plt

ctx = ["(32K, 1K)", "(32K, 4K)", "(32K, 8K)", "(32K, 32K)", "(32K, 65K)"]
x = np.arange(len(ctx))  # 真正用于画图的位置：0,1,2,...

# ====================== 示例数据，按需替换 ======================
bf16_qk   = np.array([2662, 2936, 2994, 3019, 3006])
bf16_txl  = np.array([1167, 2748, 3002, 3024, 3039])
bf16_torch = np.array([2042, 1136, 1307, 1412, 1443])
# ===============================================================

methods = ["Quack", "Txl", "Torch"]

colors = {
    "Quack": "#f1c40f",
    "Txl":           "#e74c3c",
    "Torch":         "#8e44ad",
}

data = {
    ("Softmax, BF16"): {
        "Quack": bf16_qk,
        "Txl":           bf16_txl,
        "Torch":         bf16_torch,
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
        if m == "Quack":
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
    "Softmax, BF16",
    data[("Softmax, BF16")],
    ylim=(0, 3500),
    yticks=[0, 500, 1000, 1500, 2000, 2500, 3000, 3500],
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

ax.set_ylabel("Memory Bandwidth (GB/s)")
ax.set_xlabel("(M, N): (Batch size, Reduction dim)")

plt.subplots_adjust(top=0.82)

plt.show()

# b=132, s_q=2, h_q=128, h_kv=1, d=576, dv=512, causal=False, dtype=torch.float16
