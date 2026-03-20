import os
from typing import Dict, List
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
import glob

from src import acousticFeature
from src import linguisticFeature

import parselmouth

def get_audio_files(audio_path:str) -> dict[str, List[str]]:
    """Load all audio files from ADReSSo structure
    """

    audio_files = {
        "ad": sorted((Path(audio_path) / "ad").glob('*.wav')),
        "cn": sorted((Path(audio_path) / "cn").glob('*.wav')),
    }

    print(f"Found {len(audio_files['ad'])} AD files")
    print(f"Found {len(audio_files['cn'])} CN files")
    
    return audio_files


# Runing on each segment and calculate statistics
def process_acoustic_features(audio_path:str, diarization_segment_path:str, transcript_segment_path:str) -> tuple[dict, dict]:
    """
    Extracts and aggregates acoustic features strictly from PAR segments in csv
    """

    # Load audio file
    full_sound = parselmouth.Sound(audio_path)

    # Check matching patient id
    df_original_transcript = pd.read_csv(transcript_segment_path)
    if "files_id" in df_original_transcript.columns:
        patient_id = Path(audio_path).stem
        df_transcript = df_original_transcript[df_original_transcript["files_id"] == patient_id]

    transcript = " ".join(df_transcript["transcript"].dropna().astype(str))

    # Load the diarization csv
    df_segment = pd.read_csv(diarization_segment_path)
    par_segments = df_segment[df_segment["speaker"] == "PAR"].copy()

    segment_features_list = []

    # Iterate over each PAR segment
    for index, row in par_segments.iterrows():
        start_time = row["begin"]/1000.0
        end_time = row["end"]/1000.0

        if start_time >= end_time:
            continue
        
        try:
            segment_sound = full_sound.extract_part(start_time, end_time)
        except Exception as e:
            print(f"Error extracting segment {index}: {e}")
            continue
        
        # Extract feature for this segment
        intensity_attrs, _ = acousticFeature.get_intensity_attributes(segment_sound)
        pitch_attrs, _ = acousticFeature.get_pitch_attributes(segment_sound)
        jitter_attrs = acousticFeature.get_local_jitter(segment_sound)
        shimmer_attrs = acousticFeature.get_local_shimmer(segment_sound)
        spectrum_attrs, _ = acousticFeature.get_spectrum_attributes(segment_sound)
        formant_attrs, _ = acousticFeature.get_formant_attributes(segment_sound)
        speaking_rate = acousticFeature.get_speaking_rate(audio_path, transcript)


        # Combine to dict
        segment_features = {
            "segment_id": index,
            "start_time": start_time,
            "end_time": end_time,
            **intensity_attrs,
            **pitch_attrs,
            "jitter_local": jitter_attrs,
            "shimmer_local": shimmer_attrs,
            "speaking_rate": speaking_rate,
            **spectrum_attrs,
            **formant_attrs,
        }

        segment_features_list.append(segment_features)

    # Convert to DataFrame
    df_segment_features = pd.DataFrame(segment_features_list)

    if df_segment_features.empty:
        patient_profile = pd.Series(dtype=float)
    else:
        patient_profile = df_segment_features.drop(columns=["segment_id", "start_time", "end_time"], errors="ignore").agg(["mean", "std"]).unstack()

    return df_segment_features, patient_profile


def process_linguistic_features(whisper_transcript_path:str, patient_id:str, lang:str="en") -> dict:
    """
    Extract linguistic features from transcript csv
    """

    # Check matching patient id
    df_whisper_transcript = pd.read_csv(whisper_transcript_path)
    if "files_id" in df_whisper_transcript.columns:
        df_whisper_transcript = df_whisper_transcript[df_whisper_transcript["files_id"] == patient_id]
        
    # Extract the string from the first row, or join them if there are multiple
    whisper_transcript = " ".join(df_whisper_transcript["transcript"].dropna().astype(str))

    # Feature
    cttr, brunet, std_entropy, pidensity = linguisticFeature.lexical_richness(whisper_transcript, lang=lang)
    pos_tagged_data, polarity, subjectivity = linguisticFeature.pos_polarity_subjectivity(whisper_transcript, lang=lang)
    tag_count = linguisticFeature.tag_count(pos_tagged_data)
    pos_rate = linguisticFeature.evaluate_pos_rate(tag_count)
    content_density = tag_count["content_density"]
    open_class_words = tag_count["open_class_words"]
    closed_class_words = tag_count["closed_class_words"]
    disfluency_count = linguisticFeature.count_disfluency(whisper_transcript, lang=lang)
    person_rate, spatial_rate, temporal_rate = linguisticFeature.evaluate_deixis(whisper_transcript, lang=lang)
    dale_chall, flesch, coleman_liau_index, r_time, syllables = linguisticFeature.evaluate_readability(whisper_transcript)


    result = {
        "patient_id": patient_id,
        "lang": lang,
        "cttr": cttr,
        "brunet": brunet,
        "std_entropy": std_entropy,
        "pidensity": pidensity,
        # "pos_tagged_data": pos_tagged_data,
        "content_density": content_density,
        "open_class_words": open_class_words,
        "closed_class_words": closed_class_words,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "pos_rate": pos_rate,
        "disfluency_count": disfluency_count,
        "person_rate": person_rate,
        "spatial_rate": spatial_rate,
        "temporal_rate": temporal_rate,
        "dale_chall": dale_chall,
        "flesch": flesch,
        "coleman_liau_index": coleman_liau_index,
        "r_time": r_time,
        "syllables": syllables,
    }

    return result