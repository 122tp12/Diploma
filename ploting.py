import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

FOLDER_NAME = '_for_ploting'
X_AXIS_COL = 'epoch'
OUTPUT_FILENAME = 'combined_comparison.png'

def generate_combined_plot():
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, FOLDER_NAME)

    if not os.path.exists(target_dir):
        print(f"Error: Folder '{FOLDER_NAME}' not found.")
        return

    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print("No .csv files found.")
        return

    try:
        first_df = pd.read_csv(csv_files[0])
        available_metrics = [c for c in first_df.columns if c != X_AXIS_COL]
        
        print(f"Found {len(csv_files)} files to compare: {[os.path.basename(f) for f in csv_files]}")
        print("\n--- Available Metrics ---")
        print(", ".join(available_metrics))
        print("-------------------------")

    except Exception as e:
        print(f"Error reading structure: {e}")
        return

    print("Enter metrics to compare (e.g., 'val_loss val_acc'):")
    user_input = input("> ").strip()
    
    if not user_input:
        selected_metrics = available_metrics
        print("Plotting ALL metrics.")
    else:
        selected_metrics = [m for m in user_input.split() if m in available_metrics]

    if not selected_metrics:
        print("No valid metrics selected.")
        return

    plt.style.use('ggplot')
    
    fig, axes = plt.subplots(len(selected_metrics), 1, figsize=(12, 5 * len(selected_metrics)), sharex=True)
    
    if len(selected_metrics) == 1:
        axes = [axes]

    print("\nGenerating combined plot...")

    for i, metric in enumerate(selected_metrics):
        ax = axes[i]
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                df = df.iloc[:]  # Skip first 40 epochs
                filename = os.path.basename(file_path)
                
                if metric in df.columns:
                    ax.plot(df[X_AXIS_COL], df[metric], label=filename, marker='.', linewidth=2)
                else:
                    print(f"Warning: {filename} missing column '{metric}'")
            except:
                pass

        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'Comparison: {metric}')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True)

    axes[-1].set_xlabel(X_AXIS_COL, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(target_dir, OUTPUT_FILENAME)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Success! Combined plot saved to: {save_path}")

if __name__ == "__main__":
    generate_combined_plot()