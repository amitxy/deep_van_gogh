import os
import time
import sys

ROOT = './data/dataset'
DATASET_DIR = 'data/dataset/Post_Impressionism'
CSV_PATH = 'data/dataset/classes.csv'
OPTIMIZED_DIR = './data/optimized'


def show_optimization_progress(current_size, total_size):
    sys.stdout.flush()
    percent = (current_size / total_size) * 100
    if percent == 100:
        print(f"\rOptimizing dataset... {percent:.2f}%")
    else:
        sys.stdout.write(f"\rOptimizing dataset... {percent:.2f}%")