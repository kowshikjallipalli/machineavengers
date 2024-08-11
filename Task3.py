import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

def read_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    print("Data preview:\n", df.head())
    
    if df.shape[1] < 2:
        raise ValueError("CSV file must contain at least two columns of data.")
    
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    return x, y

def fit_and_complete_curve(x, y):
    tck_x = splrep(np.arange(len(x)), x, s=0)
    tck_y = splrep(np.arange(len(y)), y, s=0)
    
    t_new = np.linspace(0, len(x) - 1, 100)
    x_new = splev(t_new, tck_x)
    y_new = splev(t_new, tck_y)
    
    return x_new, y_new

def plot_and_save_curve(x_visible, y_visible, x_completed, y_completed, output_file):
    plt.figure(figsize=(10, 6))
    plt.plot(x_visible, y_visible, 'o', label='Visible Points', color='red')
    plt.plot(x_completed, y_completed, '-', label='Completed Curve', color='blue')
    plt.title('Curve Completion Using Spline Interpolation')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()

def process_csv(file_path, output_file):
    x, y = read_csv(file_path)
    x_completed, y_completed = fit_and_complete_curve(x, y)
    plot_and_save_curve(x, y, x_completed, y_completed, output_file)

if __name__ == "__main__":
    file_type = input("Enter the file type (csv/svg): ").strip().lower()
    input_file = input("Enter the path to the input file: ").strip()
    output_file = input("Enter the path to the output file: ").strip()

    if file_type == 'csv':
        process_csv(input_file, output_file)
    else:
        print("Unsupported file type. Please choose 'csv'.")

# How to use this:
# 1. Ensure you have a CSV file with at least two columns of numeric data (x and y coordinates).
# 2. Create an 'output' folder in your working directory if it doesn't already exist.
# 3. Create an empty .png file in the 'output' folder where the completed curve will be saved.
# 4. Run the script and provide the path to your input CSV file and the path to the output .png file when prompted.
