import os
from typing import Dict, List

import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm

def get_audio_files(audio_path:str) -> dict[str, List[str]]:
    """Load all audio files from ADReSSo structure
    """

    audio_files = {
        "ad": sorted((audio_path/"ad").glob("*.wav")),
        "cn": sorted((audio_path/"cn").glob("*.wav")),
    }

    print(f"Found {len(audio_files["ad"])} AD files")
    print(f"Found {len(audio_files["cn"])} CN files")
    
    return audio_files