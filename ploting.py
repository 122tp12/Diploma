import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- CONFIGURATION ---
FOLDER_NAME = '_for_ploting'
X_AXIS_COL = 'epoch'
OUTPUT_FILENAME = 'combined_comparison.png'
# ---------------------

def generate_combined_plot():
    # 1. Setup paths
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, FOLDER_NAME)

    if not os.path.exists(target_dir):
        print(f"Error: Folder '{FOLDER_NAME}' not found.")
        return

    csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
    if not csv_files:
        print("No .csv files found.")
        return

    # 2. Read first file to get available metrics
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

    # 3. User Input
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

    # 4. Generate ONE figure with subplots for each chosen metric
    plt.style.use('ggplot')
    
    # Create subplots: One row per metric
    fig, axes = plt.subplots(len(selected_metrics), 1, figsize=(12, 5 * len(selected_metrics)), sharex=True)
    
    # Ensure axes is iterable even if there is only 1 metric
    if len(selected_metrics) == 1:
        axes = [axes]

    print("\nGenerating combined plot...")

    # Cycle through the metrics user chose (e.g., first plot val_loss, then plot val_acc)
    for i, metric in enumerate(selected_metrics):
        ax = axes[i]
        
        # For this specific metric, loop through ALL files and add a line
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                filename = os.path.basename(file_path)
                
                # Check if this specific file actually has the column
                if metric in df.columns:
                    ax.plot(df[X_AXIS_COL], df[metric], label=filename, marker='.', linewidth=2)
                else:
                    print(f"Warning: {filename} missing column '{metric}'")
            except:
                pass

        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'Comparison: {metric}')
        ax.legend(loc='best', fontsize='small') # Show filenames in legend
        ax.grid(True)

    # Set X-label on the bottom-most plot only
    axes[-1].set_xlabel(X_AXIS_COL, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(target_dir, OUTPUT_FILENAME)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Success! Combined plot saved to: {save_path}")

if __name__ == "__main__":
    generate_combined_plot()