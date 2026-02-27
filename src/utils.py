import os
from typing import Dict, List
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm
import glob

def get_audio_files(audio_path:str) -> dict[str, List[str]]:
    """Load all audio files from ADReSSo structure
    """

    audio_files = {
        "ad": sorted((Path(audio_path) / "ad").glob('*.wav')),
        "cn": sorted((Path(audio_path) / "cn").glob('*.wav')),
    }

    print(f"Found {len(audio_files["ad"])} AD files")
    print(f"Found {len(audio_files["cn"])} CN files")
    
    return audio_files