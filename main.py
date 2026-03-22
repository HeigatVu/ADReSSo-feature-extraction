import os
import src.utils as utils
from pathlib import Path
import pandas as pd
import torch

def main(transcript:bool=True, feature:bool=True) -> str:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Transcribe audio files
    if transcript:
        # Get files
        path_config = utils.load_yaml("src/config/path.yaml")
        audio_train_files = utils.get_files(path_config["train"]["AUDIO_TRAIN_PATH"], data_type="train")
        audio_test_files = utils.get_files(path_config["test"]["AUDIO_TEST_PATH"], data_type="test")

        model_config = utils.load_yaml("src/config/model.yaml")
        utils.transcribe_audio_files(audio_train_files, 
                                    path_config["train"]["MMSE_DIAG_TRAIN_PATH"], 
                                    path_config["train"]["CSV_SEGMENT_TRAIN_PATH"], 
                                    data_type="train", 
                                    multipleGPU=model_config["whisper"]["MULTIPLE_GPU"], 
                                    model_name=model_config["whisper"]["MODEL_NAME"], 
                                    batch_size=model_config["whisper"]["BATCH_SIZE"], 
                                    device=DEVICE, 
                                    output_path=path_config["TRANSCRIPT_PATH"])
        utils.transcribe_audio_files(audio_test_files, 
                                    path_config["test"]["MMSE_DIAG_TEST_PATH"], 
                                    path_config["test"]["CSV_SEGMENT_TEST_PATH"], 
                                    data_type="test", 
                                    multipleGPU=model_config["whisper"]["MULTIPLE_GPU"], 
                                    model_name=model_config["whisper"]["MODEL_NAME"], 
                                    batch_size=model_config["whisper"]["BATCH_SIZE"], 
                                    device=DEVICE, 
                                    output_path=path_config["TRANSCRIPT_PATH"])
        return f"finish transcript train and test set"
    
    if feature:
        pass
        return f"finish feature extraction train and test set"

if __name__ == "__main__":
    # BASE_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration"

    # OUTPUT_FEATURE_PATH = f"{BASE_PATH}/output/features"

    # # Train paths
    # CSV_SEGMENT_TRAIN_PATH = f"{BASE_PATH}/data/diagnosis/train/segmentation"
    # AUDIO_TRAIN_PATH = f"{BASE_PATH}/data/diagnosis/train/audio"
    # MMSE_DIAG_TRAIN_PATH = f"{BASE_PATH}/data/diagnosis/train/adresso-train-mmse-scores.csv"

    # # Test paths
    # CSV_SEGMENT_TEST_PATH = f"{BASE_PATH}/data/diagnosis/test-dist/segmentation"
    # AUDIO_TEST_PATH = f"{BASE_PATH}/data/diagnosis/test-dist/audio"
    # MMSE_DIAG_TEST_PATH = f"{BASE_PATH}/data/diagnosis/test-dist/adresso-test-mmse-scores.csv"

    # # Extract transcripts
    # TRANSCRIPT_PATH = f"{BASE_PATH}/output/transcripts"
    # # Model setup
    # MODEL_NAME = "openai/whisper-large-v3"
    # BATCH_SIZE = 8


    main(transcript=True, feature=True)