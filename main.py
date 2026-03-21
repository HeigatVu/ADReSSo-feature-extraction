import os
import src.utils as utils
from pathlib import Path

if __name__ == "__main__":
    BASE_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration"
    TRANSCRIPT_PATH = f"{BASE_PATH}/output/transcripts"
    CSV_PATH = f"{BASE_PATH}/data/diagnosis/train/segmentation"
    OUTPUT_FEATURE_PATH = f"{BASE_PATH}/output/features"

    utils.feature_extraction(OUTPUT_FEATURE_PATH, TRANSCRIPT_PATH, CSV_PATH)