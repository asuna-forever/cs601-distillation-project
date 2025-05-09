# scripts/plot_loss.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_loss_curve(csv_path, output_image_path, window_size=40):
    """
    Plots the training loss curves (total, CE, KL) from a CSV file
    and saves the plot to an image file. Optionally applies a smoothing window.
    """

    if not os.path.exists(csv_path):
        print(f"Error: Loss CSV file not found: {csv_path}")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} loss records from {csv_path}.")

        # Check if necessary columns exist
        required_columns = ['step', 'total_loss', 'loss_ce', 'loss_kl']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV file is missing necessary columns. Required: {required_columns}")
            print(f"  Actual columns included: {df.columns.tolist()}")
            return

        # Create the plot
        plt.figure(figsize=(12, 8)) # You can adjust the figure size, e.g., increase height

        # Apply moving window smoothing
        if window_size and window_size > 1:
            print(f"Applying a moving window of size {window_size} for smoothing.")
            # Create new columns to store smoothed data to avoid modifying original data (if needed)
            df['total_loss_smooth'] = df['total_loss'].rolling(window=window_size, center=True, min_periods=1).mean()
            df['loss_ce_smooth'] = df['loss_ce'].rolling(window=window_size, center=True, min_periods=1).mean()
            df['loss_kl_smooth'] = df['loss_kl'].rolling(window=window_size, center=True, min_periods=1).mean()
            # center=True means the label of the window is the center of the window
            # min_periods=1 ensures that an average is calculated even at the beginning of the window (fewer data points than window_size)

            total_loss_to_plot = df['total_loss_smooth']
            loss_ce_to_plot = df['loss_ce_smooth']
            loss_kl_to_plot = df['loss_kl_smooth']
            plot_suffix = f" (Smoothed, Window={window_size})"
        else:
            print("No smoothing applied (window_size <= 1 or None).")
            total_loss_to_plot = df['total_loss']
            loss_ce_to_plot = df['loss_ce']
            loss_kl_to_plot = df['loss_kl']
            plot_suffix = ""


        # Plot Total Loss
        plt.plot(df['step'], total_loss_to_plot, label=f'Total Loss{plot_suffix}', alpha=0.9, linewidth=2)
        # Plot Cross-Entropy Loss
        plt.plot(df['step'], loss_ce_to_plot, label=f'Cross-Entropy Loss{plot_suffix}', alpha=0.7, linestyle='--')
        # Plot KL Loss
        plt.plot(df['step'], loss_kl_to_plot, label=f'KL divergence Loss{plot_suffix}', alpha=0.7, linestyle=':')

        # (Optional) Plot raw data for comparison
        # if window_size and window_size > 1:
        #     plt.plot(df['step'], df['total_loss'], label='Total Loss (Raw)', alpha=0.3, linewidth=1, color='lightblue')
        #     plt.plot(df['step'], df['loss_ce'], label='Cross-Entropy Loss (Raw)', alpha=0.3, linestyle='--', color='lightcoral')
        #     plt.plot(df['step'], df['loss_kl'], label='KL divergence Loss (Raw)', alpha=0.3, linestyle=':', color='lightgreen')


        # Add chart elements
        plt.title(f'Training Loss Curve{plot_suffix}') # Chart title
        plt.xlabel('Training Steps') # X-axis label
        plt.ylabel('Loss Value') # Y-axis label
        plt.legend() # Display legend
        plt.grid(True, linestyle='--', alpha=0.6) # Display grid and adjust style
        plt.yscale('log') # *** Using a logarithmic scale might make it easier to observe initial changes ***
        # Or set a reasonable Y-axis range if the approximate range of loss is known
        # plt.ylim(bottom=0, top=max(df['total_loss'].max(), df['loss_ce'].max(), df['loss_kl'].max()) * 1.1) # Dynamic range + 10% margin

        # Improve layout
        plt.tight_layout()

        # Save the chart
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight') # Save as an image file, set resolution, bbox_inches ensures labels are not cropped
        print(f"Loss curve plot saved to: {output_image_path}")

        # (Optional) Display the chart
        # plt.show()

    except Exception as e:
        print(f"Error plotting or saving the chart: {e}")

# --- Command-line argument parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training Loss curve")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./DistilQwen3/training_losses.csv", # Default path, consistent with train.py save location
        help="Path to the CSV file containing training loss data"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="./distilgpt2_from_qwen/training_loss_curve.png", # Default save path
        help="File path to save the loss curve plot"
    )
    parser.add_argument(
        "--window_size", # New command-line argument
        type=int,
        default=40, # Default moving window size
        help="Moving window size for smoothing the curve. Set to 1 for no smoothing."
    )
    args = parser.parse_args()

    # Call the plotting function
    plot_loss_curve(args.csv_path, args.output_image, args.window_size)