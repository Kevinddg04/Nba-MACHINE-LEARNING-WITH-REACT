import os
import pandas as pd

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_83cf42e46920a6011c69ce94cde5bbee'
    api = KaggleApi()
    api.authenticate()
    print("Descargando data temp...")
    api.dataset_download_file('eoinamoore/historical-nba-data-and-player-box-scores', 'TeamStatistics.csv', path='.', force=True)
    import zipfile
    if os.path.exists("TeamStatistics.csv.zip"):
        with zipfile.ZipFile("TeamStatistics.csv.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    
    df = pd.read_csv("TeamStatistics.csv", nrows=1)
    print("COLUMNAS ENCONTRADAS:")
    print(list(df.columns))
except Exception as e:
    print(f"Error: {e}")
