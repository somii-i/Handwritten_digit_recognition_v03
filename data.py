import pandas as pd
from config import Config
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data():
    Config.create_dirs()
    
    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset
    api.competition_download_files('digit-recognizer', path=Config.DATA_PATH)
    
    # Unzip the files
    import zipfile
    with zipfile.ZipFile(Config.DATA_PATH + 'digit-recognizer.zip', 'r') as zip_ref:
        zip_ref.extractall(Config.DATA_PATH)
    
    print("Data downloaded and extracted successfully")

if __name__ == '__main__':
    download_data()