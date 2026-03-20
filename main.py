import os
import src.utils as utils
import glob
import pandas as pd
from pathlib import Path
import tqdm

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
    OUTPUT_FEATURE_PATH = f"{BASE_PATH}/output/features"

    output_feature_file = Path(OUTPUT_FEATURE_PATH) / f"adresso_features.csv"
    Path(OUTPUT_FEATURE_PATH).mkdir(parents=True, exist_ok=True)
    
    transcript_files = glob.glob(TRANSCRIPT_PATH + "/*.csv")[0]
    # Sample information and data
    df_sample_info = pd.read_csv(transcript_files)
    transcript = df_sample_info["transcript"]
    patient_id = df_sample_info["files_id"]
    audio_path = df_sample_info["audio_path"]
    
    # Diazrization of samples
    diagnosis_list = df_sample_info["diagnosis"]

    df_feature = pd.DataFrame()
    for i in tqdm.tqdm(range(len(df_sample_info))):
        patient = patient_id[i]
        diag = diagnosis_list[i]
        segment_file = f"{CSV_PATH}/{diag}/{patient}.csv"
        
        # Ensure we skip if there are no PAR segments in the CSV
        if os.path.exists(segment_file):
            df_segment = pd.read_csv(segment_file)
            if "PAR" not in df_segment["speaker"].values:
                continue
            
        df_feature = pd.concat([df_feature, process_feature(audio_path[i], segment_file, transcript_files, patient, lang="en")], ignore_index=True)
    df_feature.to_csv(output_feature_file, index=False)