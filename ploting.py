import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Додано для генерації індексів
import os
import glob

FOLDER_NAME = '_for_ploting'
X_AXIS_COL = 'epoch'
OUTPUT_FILENAME = 'combined_comparison.png'
WINDOW_SIZE = 1     # Кількість епох для усереднення в одній точці

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
                df = df.iloc[:50]  
                filename = os.path.basename(file_path)
                
                if metric in df.columns:
                    
                    # 1. Розбиваємо дані на групи по WINDOW_SIZE (5) елементів
                    group_ids = np.arange(len(df)) // WINDOW_SIZE
                    
                    # 2. Обчислюємо середнє значення та стандартне відхилення для кожної групи
                    df_mean = df.groupby(group_ids).mean()
                    df_std = df.groupby(group_ids).std()
                    
                    x = df_mean[X_AXIS_COL]
                    y = df_mean[metric]
                    
                    # Заповнюємо можливі NaN нулями (виникає, якщо в останній групі залишився лише 1 елемент)
                    y_err = df_std[metric].fillna(0) 
                    
                    # 3. Будуємо основну лінію середнього значення. 
                    # Зберігаємо об'єкт [0], щоб отримати автоматично призначений колір
                    line = ax.plot(x, y, label=filename, marker='.', linewidth=2)[0]
                    
                    # 4. Будуємо напівпрозоре (alpha=0.2) відхилення тим самим кольором
                    ax.fill_between(x, y - y_err, y + y_err, color=line.get_color(), alpha=0.2)

                else:
                    print(f"Warning: {filename} missing column '{metric}'")
            except Exception as e:
                print(f"Error plotting {filename}: {e}")

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