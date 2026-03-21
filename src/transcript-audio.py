import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm

from src import utils

BASE_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration"
# # Run medium model
# MODEL_NAME = "openai/whisper-medium.en"
# BATCH_SIZE = 8
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRAIN_PATH_DATA = f"{BASE_PATH}/data/diagnosis/train"
# TRAIN_AUDIO_PATH = f"{TRAIN_PATH_DATA}/audio"
# OUTPUT_PATH = f"{BASE_PATH}/output/transcripts"
# os.makedirs(OUTPUT_PATH, exist_ok=True)

# print(f"Using device: {DEVICE}")
# print(f"Model: {MODEL_NAME}")

# Run large model
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH_DATA = f"{BASE_PATH}/data/diagnosis/train"
TRAIN_AUDIO_PATH = f"{TRAIN_PATH_DATA}/audio"
OUTPUT_PATH = f"{BASE_PATH}/output/transcripts"
MMSE_PATH = "/mnt/data_lab513/ducvu/ADReSSo/ADReSSo-feature-extration/data/diagnosis/train/adresso-train-mmse-scores.csv"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Model: {MODEL_NAME}")

audio_files_dict = utils.get_audio_files(TRAIN_AUDIO_PATH)
print(audio_files_dict)

# Run medium model on one GPU
transcriber = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    model_kwargs={
        "attn_implementation": "sdpa"  # Scaled dot-product attention (faster)
    }
)

transcriber.model.config.forced_decoder_ids = transcriber.tokenizer.get_decoder_prompt_ids(
    language="english",
    task="transcribe"
)


# # processor = AutoProcessor.from_pretrained(MODEL_NAME)
# torch_type=torch.float16 if torch.cuda.is_available() else torch.float32

# # Create pipeline
# transcriber = pipeline(
#     task="automatic-speech-recognition",
#     batch_size=BATCH_SIZE,
#     model=MODEL_NAME,
#     torch_dtype=torch_type,
#     device_map="auto", # Using this to multiGPU
#     generate_kwargs={
#         "language": "english"
#     },
#     model_kwargs={
#         "attn_implementation": "sdpa"  # Scaled dot-product attention (faster)
#     }
# )

def transcribe_audio_files(
    audio_files:Dict[str, List[Path]],
    transcriber, # model
) -> pd.DataFrame:
    """ Transcribe audio files without diarization
    """

    results = []
    df_data = pd.read_csv(MMSE_PATH)

    for diagnosis, files in audio_files.items():
        for audio_file in tqdm(files, desc=f"{diagnosis.upper()}"):
            output = transcriber(
                str(audio_file),
                return_timestamps=True,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "en",
                    "return_timestamps": True,
                    "num_beams": 5,
                }
            )

            # Handle different output formats
            if isinstance(output, dict):
                if "text" in output:
                    transcript = output["text"].strip()
                elif "chunks" in output:
                    transcript = " ".join([chunk["text"] for chunk in output["chunks"]]).strip()
                else:
                    transcript = ""
            else:
                transcript = str(output).strip()

            patient_row = df_data[df_data["adressfname"] == audio_file.stem]

            results.append({
                "files_id": audio_file.stem,
                "mmse_score": patient_row["mmse"].values[0],
                "audio_path": str(audio_file),
                "diagnosis": diagnosis,
                "transcript": transcript,
                })
    return pd.DataFrame(results)

df_transcripts = transcribe_audio_files(
    audio_files=audio_files_dict,
    transcriber=transcriber,
)

print(f"\nTotal transcription: {len(df_transcripts)}")
print(f"\nSample transcripts: {df_transcripts.head()}")
print(df_transcripts["transcript"])

# Save to csv
output_file = Path(OUTPUT_PATH) / f"adresso_transcripts_{MODEL_NAME.split('/')[-1]}.csv"
df_transcripts.to_csv(output_file, index=False)