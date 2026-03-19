import os
import src.utils as utils
import glob
import pandas as pd
from pathlib import Path

def process_feature(audio_path:str, csv_segment_path:str, transcript_path:str, patient_id:str, lang:str="en") -> pd.DataFrame:
    processed_linguistic_feature = utils.process_linguistic_features(transcript_path, patient_id, lang=lang)
    processed_acoustic_feature = utils.process_acoustic_features(audio_path, csv_segment_path, transcript_path)
    
    # Flatten linguistic features
    flat_ling = {}
    for k, v in processed_linguistic_feature.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_ling[f"{k}_{sub_k}"] = sub_v
        else:
            flat_ling[k] = v
            
    # Flatten acoustic features
    patient_profile_series = processed_acoustic_feature[1]
    flat_acoustic = {}
    for (feat, stat), val in patient_profile_series.items():
        flat_acoustic[f"{feat}_{stat}"] = val
    
    # Combine both into a single dictionary and convert to DataFrame
    combined_features = {**flat_ling, **flat_acoustic}
    df_combined = pd.DataFrame([combined_features])


    
    return df_combined



if __name__ == "__main__":
    BASE_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration"
    TRANSCRIPT_PATH = f"{BASE_PATH}/output/transcripts"
    CSV_PATH = f"{BASE_PATH}/data/diagnosis/train/segmentation"
    
    transcript_files = glob.glob(TRANSCRIPT_PATH + "/*.csv")[0]
    df_patient_info = pd.read_csv(transcript_files)
    transcript = df_patient_info["transcript"]
    patient_id = df_patient_info["files_id"]
    audio_path = df_patient_info["audio_path"]
    csv_segment_path = glob.glob(CSV_PATH + "/ad/*.csv") + glob.glob(CSV_PATH + "/cn/*.csv")
    print(len(csv_segment_path))



    # df_feature = process_feature(AUDIO_PATH, CSV_PATH, transcript_files, test_patient_id, lang="en")

    # OUTPUT_PATH = f"{BASE_PATH}/output/features"
    # output_file = Path(OUTPUT_PATH) / f"adresso_features.csv"
    
    # # Add this line to create the directory if it doesn't exist
    # Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    
    # df_feature.to_csv(output_file, index=False)
