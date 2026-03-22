from typing import Dict, List
from pathlib import Path
from omegaconf import OmegaConf

def load_yaml(yaml_path:str) -> dict:
    """Load YAML file and set BASE_PATH to project root
    """
    conf = OmegaConf.load(yaml_path)
    if "BASE_PATH" in conf:
        project_root = str(Path(__file__).parent.parent.absolute())
        conf.BASE_PATH = project_root
    return conf


def get_files(audio_path:str, 
                    data_type:str="train") -> dict[str, List[str]]:
    """Load all audio files from ADReSSo structure
    """
    if data_type == "train":
        audio_files = {
            "ad": sorted((Path(audio_path) / "ad").glob('*.*')),
            "cn": sorted((Path(audio_path) / "cn").glob('*.*')),
        }
    elif data_type == "test":
        audio_files = {
            "audio": sorted((Path(audio_path)).glob('*.*')),
        }
    else:
        raise ValueError("Invalid data type. Must be 'train' or 'test'.")
    
    return audio_files

# # Split data into train, validation, and test sets
# def train_val_test_split(df_train:pd.DataFrame, 
#                         test_size:float=0.2, 
#                         random_state:int=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """ Split data into train and validation datasets
#     """
    
#     df_x_train, df_x_val, df_y_train, df_y_val = train_test_split(df_train, test_size=test_size, 
#                                                                 random_state=random_state)

#     return df_x_train, df_x_val, df_y_train, df_y_val

