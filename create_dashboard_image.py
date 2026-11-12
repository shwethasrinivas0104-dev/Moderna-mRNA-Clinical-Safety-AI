from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. ROC Curve
roc = FIGURE_DIR / "roc_curve.png"
if roc.exists():
    axes[0].imshow(mpimg.imread(roc))
    axes[0].set_title("Week 1 - ROC Curve")
else:
    axes[0].text(0.5, 0.5, "ROC curve not found", ha="center")
axes[0].axis("off")

# 2. Attention Heatmap
heatmap = FIGURE_DIR / "attention_heatmap.png"
if heatmap.exists():
    axes[1].imshow(mpimg.imread(heatmap))
    axes[1].set_title("Week 2 - Attention Heatmap")
else:
    axes[1].text(0.5, 0.5, "Heatmap not found", ha="center")
axes[1].axis("off")

# 3. RAG Retrieval Latency
csv_path = PROJECT_DIR / "outputs" / "rag_evaluation.csv"

if csv_path.exists():
    df = pd.read_csv(csv_path)
    latency_col = next(
        (col for col in df.columns
         if "latency" in col.lower() or "time" in col.lower()),
        None
    )

    if latency_col:
        axes[2].bar(range(len(df)), df[latency_col])
        axes[2].set_title("Week 3 - Retrieval Latency")
        axes[2].set_xlabel("Test Question")
        axes[2].set_ylabel("Seconds")
    else:
        axes[2].text(0.5, 0.5, "Latency column not found", ha="center")
else:
    axes[2].text(0.5, 0.5, "RAG results not found", ha="center")

fig.suptitle("mRNA-VIP Project Dashboard", fontsize=18)
plt.tight_layout()

output_path = FIGURE_DIR / "dashboard_summary.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print("Dashboard PNG saved successfully!")
print(output_path)

