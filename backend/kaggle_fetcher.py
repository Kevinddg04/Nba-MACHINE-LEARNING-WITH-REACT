"""
kaggle_fetcher.py
=================
Reemplaza la descarga manual del CSV de Kaggle.
Descarga y actualiza TeamStatistics.csv automáticamente.

SETUP (una sola vez):
    1. pip install kaggle pandas
    2. Ir a https://www.kaggle.com/settings → API → "Create New Token"
    3. Mueve el archivo kaggle.json descargado a:
         - Windows: C:/Users/<tu_usuario>/.kaggle/kaggle.json
         - Mac/Linux: ~/.kaggle/kaggle.json
    4. Edita KAGGLE_DATASET abajo con el slug de tu dataset

USO:
    python kaggle_fetcher.py              # descarga/actualiza el CSV
    python kaggle_fetcher.py --force      # sobreescribe sin preguntar
"""

import os
import shutil
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  CONFIGURA AQUÍ TU DATASET
#  Formato: "usuario/nombre-del-dataset"
#  Ejemplo: si la URL de Kaggle es
#    kaggle.com/datasets/johndoe/nba-team-statistics
#  entonces KAGGLE_DATASET = "johndoe/nba-team-statistics"
# ─────────────────────────────────────────────────────────────
KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"

OUTPUT_CSV     = "TeamStatistics.csv"       # nombre que usa tu notebook
DOWNLOAD_DIR   = Path("kaggle_downloads")   # carpeta temporal


def check_kaggle_credentials():
    """Verifica que las credenciales de Kaggle estén configuradas (archivo o variables de entorno)."""
    # Chequeo para formato nuevo de token único
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    # Chequeo para formato viejo de usuario/clave
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ No se encontraron variables de entorno ni ~/.kaggle/kaggle.json")
        print("   Pasos para local:")
        print("   1. Ve a https://www.kaggle.com/settings → API → Create New Token")
        print("   2. Mueve el archivo descargado a ~/.kaggle/kaggle.json")
        print("   Pasos para Render/Nube:")
        print("   Agrega KAGGLE_USERNAME y KAGGLE_KEY en tu sección de Settings > Environment Variables.")
        return False
    return True


def download_dataset(force: bool = False) -> Path:
    """
    Descarga el dataset de Kaggle a DOWNLOAD_DIR.
    Retorna la ruta al CSV descargado.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("Instala la librería: pip install kaggle")

    if not check_kaggle_credentials():
        raise RuntimeError("Configura las credenciales de Kaggle primero.")

    if KAGGLE_DATASET == "TU_USUARIO/TU_DATASET":
        raise ValueError(
            "Edita KAGGLE_DATASET en kaggle_fetcher.py con el slug de tu dataset.\n"
            "Ejemplo: 'johndoe/nba-team-statistics'"
        )

    DOWNLOAD_DIR.mkdir(exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"[Kaggle] Descargando dataset: {KAGGLE_DATASET} ...")
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DOWNLOAD_DIR),
        unzip=True,
        force=force,
        quiet=False,
    )
    print(f"[Kaggle] ✅ Descarga completada en {DOWNLOAD_DIR}/")

    # Buscar games.csv dentro de la carpeta descargada
    csv_files = list(DOWNLOAD_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontró ningún CSV en {DOWNLOAD_DIR}. "
            "Verifica que el dataset contiene archivos .csv."
        )

    # Si hay varios CSVs, preferir el que tenga "Team" en el nombre o TeamStatistics.csv
    team_csvs = [f for f in csv_files if "team" in f.name.lower() or "Team" in f.name]
    chosen = team_csvs[0] if team_csvs else csv_files[0]
    print(f"[Kaggle] Usando archivo: {chosen.name}")
    return chosen


def update_local_csv(new_csv_path: Path, force: bool = False):
    """
    Combina el CSV recién descargado con el CSV local existente.
    Elimina duplicados por gameId + teamId para no perder el histórico.
    """
    new_df = pd.read_csv(new_csv_path, index_col=0, low_memory=False)
    print(f"[CSV] Dataset nuevo: {len(new_df):,} filas, {new_df.shape[1]} columnas")

    output_path = Path(OUTPUT_CSV)

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path, index_col=0, low_memory=False)
        print(f"[CSV] CSV existente: {len(existing):,} filas")

        # Combinar y deduplicar
        combined = pd.concat([existing, new_df], ignore_index=True)

        # Columnas de deduplicación (ajustar si tu dataset usa nombres distintos)
        dedup_cols = []
        if "gameId" in combined.columns and "teamId" in combined.columns:
            dedup_cols = ["gameId", "teamId"]
        elif "game_id" in combined.columns:
            dedup_cols = ["game_id"]

        if dedup_cols:
            before = len(combined)
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            after  = len(combined)
            print(f"[CSV] Deduplicados: {before - after} filas removidas")
        else:
            combined = combined.drop_duplicates()

        print(f"[CSV] Total final: {len(combined):,} filas")
    else:
        combined = new_df
        if force:
            print("[CSV] Modo --force: sobreescribiendo CSV existente")

    # Ordenar si existe gameDateTimeEst
    if "gameDateTimeEst" in combined.columns:
        combined["gameDateTimeEst"] = pd.to_datetime(
            combined["gameDateTimeEst"], errors="coerce", utc=True
        )
        combined = combined.sort_values("gameDateTimeEst").reset_index(drop=True)

    combined.to_csv(output_path)
    print(f"[CSV] ✅ Guardado: {output_path} ({len(combined):,} filas)")
    return combined


def run(force: bool = False):
    """Flujo completo: descargar → combinar → guardar."""
    print(f"\n{'='*55}")
    print(f"  NBA Kaggle Fetcher — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    try:
        csv_path = download_dataset(force=force)
        df = update_local_csv(csv_path, force=force)

        # Limpieza de archivos temporales
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        print(f"\n✅ {OUTPUT_CSV} actualizado con éxito.")
        print(f"   Filas totales: {len(df):,}")
        print(f"   Rango de fechas: {df['gameDateTimeEst'].min()} → {df['gameDateTimeEst'].max()}"
              if "gameDateTimeEst" in df.columns else "")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Kaggle Dataset Fetcher")
    parser.add_argument("--force", action="store_true",
                        help="Sobreescribe el CSV local sin combinar")
    args = parser.parse_args()
    run(force=args.force)
