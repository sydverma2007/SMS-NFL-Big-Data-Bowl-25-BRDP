import pandas as pd
import matplotlib.pyplot as plt

def plot_brdp_distribution(brdp_path="../data/brdp_results.csv"):
    df = pd.read_csv(brdp_path)
    plt.hist(df["reaction_delay_seconds"], bins=20, edgecolor="black")
    plt.title("Ball Reaction Delay Penalty Distribution")
    plt.xlabel("Reaction Delay (seconds)")
    plt.ylabel("Defender Count")
    plt.show()

if __name__ == "__main__":
    plot_brdp_distribution()
