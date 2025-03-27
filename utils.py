import os
import time
import sys

# DATA_DIR = './data/dataset'
DATA_DIR = r'D:\Users\amitr5\Documents\data'
DATASET_DIR = f'{DATA_DIR}/dataset/Post_Impressionism'
CSV_PATH = f'{DATA_DIR}/dataset/classes.csv'
OPTIMIZED_DIR = f'{DATA_DIR}/optimized/'
MODELS_DIR = f'{DATA_DIR}/models/'

def show_optimization_progress(current_size, total_size):
    sys.stdout.flush()
    percent = (current_size / total_size) * 100
    if percent == 100:
        print(f"\rOptimizing dataset... {percent:.2f}%")
    else:
        sys.stdout.write(f"\rOptimizing dataset... {percent:.2f}%")