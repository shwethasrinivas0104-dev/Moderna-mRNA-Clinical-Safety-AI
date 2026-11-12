import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def self_attention(sequence):
    # Give each RNA base a simple numerical representation
    base_values = {
        "A": [1.0, 0.0],
        "U": [0.0, 1.0],
        "G": [1.0, 1.0],
        "C": [-1.0, 0.5]
    }

    embeddings = np.array([base_values[base] for base in sequence])

    # Create Query, Key, and Value matrices
    query = embeddings
    key = embeddings
    value = embeddings

    # Calculate attention scores
    scores = query @ key.T / np.sqrt(embeddings.shape[1])

    # Convert scores into attention weights using softmax
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    attention_weights = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    # Combine information using attention weights
    output = attention_weights @ value

    return attention_weights, output


def main():
    sequence = "AUGCUA"

    attention_weights, output = self_attention(sequence)

    print("Example mRNA sequence:", sequence)
    print("\nAttention weights:")
    print(np.round(attention_weights, 3))
    print("\nAttention output:")
    print(np.round(output, 3))

    project_root = Path(__file__).resolve().parent.parent
    figures_path = project_root / "outputs" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    plt.imshow(attention_weights, cmap="viridis")
    plt.colorbar(label="Attention weight")
    plt.xticks(range(len(sequence)), list(sequence))
    plt.yticks(range(len(sequence)), list(sequence))
    plt.xlabel("RNA base being attended to")
    plt.ylabel("RNA base doing the attending")
    plt.title("Self-Attention Heatmap")
    plt.tight_layout()

    image_path = figures_path / "attention_heatmap.png"
    plt.savefig(image_path, dpi=150)
    plt.close()

    print(f"\nHeatmap saved to: {image_path}")

    weights_path = figures_path / "attention_weights.csv"
    np.savetxt(weights_path, attention_weights, delimiter=",", fmt="%.4f")
    print(f"Attention weights saved to: {weights_path}")
  
if __name__ == "__main__":
    main()

