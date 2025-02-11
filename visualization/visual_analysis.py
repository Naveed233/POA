from matplotlib import pyplot as plt
import numpy as np

def plot_metrics(optimized_weights, asset_labels):
    """Plots the optimized portfolio weights with proper asset labels and percentages."""
    
    # Ensure weights are 1D
    optimized_weights = np.array(optimized_weights).flatten()
    
    # Normalize weights to sum to 100%
    optimized_weights = optimized_weights / np.sum(optimized_weights)
    
    # Convert to percentages
    weight_percentages = optimized_weights * 100
    
    # Ensure asset labels match
    if len(asset_labels) != len(optimized_weights):
        raise ValueError(f"Mismatch between asset labels ({len(asset_labels)}) and weights ({len(optimized_weights)})")

    # Plot portfolio weights
    plt.figure(figsize=(12, 6))
    bars = plt.bar(asset_labels, weight_percentages, color='steelblue')

    # Rotate labels for clarity
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Add numerical values above bars
    for bar, weight in zip(bars, weight_percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{weight:.2f}%", ha='center', fontsize=10)

    # Titles and Labels
    plt.xlabel('Asset Type', fontsize=14)
    plt.ylabel('Portfolio Allocation (%)', fontsize=14)
    plt.title('Optimized Portfolio Weights', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Show Plot
    plt.show()
