import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def add_box(ax, text, xy, width=2.5, height=0.8, color="#AED6F1"):
    x, y = xy
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.2", 
                         linewidth=1.5, edgecolor="black", facecolor=color)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha="center", va="center", fontsize=10, fontweight="bold")

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(0, 15)
ax.set_ylim(0, 3)
ax.axis("off")

# Pipeline steps
steps = [
    "Input\nFundus Images",
    "Preprocessing\n(Resize, Normalize,\nAugment, Balance)",
    "CNN Backbone\n(ResNet50 / EfficientNet)",
    "Classification\n(Dense + Softmax)",
    "Output\nDR Grade (0–4)"
]

# Add boxes + arrows
x = 0.5
for step in steps:
    add_box(ax, step, (x, 1))
    if step != steps[-1]:
        ax.annotate("", xy=(x+2.5, 1.4), xytext=(x+3.2, 1.4),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    x += 3.2

plt.tight_layout()
plt.savefig("pipeline.png", dpi=300)
plt.show()
