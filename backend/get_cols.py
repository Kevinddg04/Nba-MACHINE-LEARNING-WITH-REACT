import os
import subprocess
import csv

try:
    subprocess.run(["pip", "install", "kagglehub"], check=True)
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_83cf42e46920a6011c69ce94cde5bbee'
    import kagglehub
    
    print("Descargando data con kagglehub...")
    path = kagglehub.dataset_download("eoinamoore/historical-nba-data-and-player-box-scores")
    
    print(f"Descargado en: {path}")
    for fname in os.listdir(path):
        if fname.endswith('TeamStatistics.csv'):
            with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                print(f"--- {fname} COLUMNAS ---")
                print(headers)
except Exception as e:
    print(f"Error grave: {e}")
