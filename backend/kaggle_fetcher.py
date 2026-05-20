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
    NUEVO METODO KAGGLE:
         - Proveer la variable de entorno `KAGGLE_API_TOKEN` con el valor `KGAT_...` en Render.
    4. KAGGLE_DATASET contiene el slug de nuestro dataset histórico.

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
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

KAGGLE_DATASET = "eoinamoore/historical-nba-data-and-player-box-scores"
OUTPUT_CSV     = str(PROJECT_ROOT / "TeamStatistics.csv")       # Nombre que usa el modelo
DOWNLOAD_DIR   = BASE_DIR / "kaggle_downloads"                  # Carpeta temporal


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
        print("   2. Configura KAGGLE_API_TOKEN como variable de sistema.")
        print("   Pasos para Render/Nube:")
        print("   Agrega KAGGLE_API_TOKEN en tu sección de Settings > Environment Variables.")
        return False
    return True


def download_dataset(force: bool = False) -> Path:
    """
    Descarga el dataset de Kaggle a la carpeta temporal.
    Retorna la ruta al CSV descargado.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError("Instala la librería: pip install kaggle")

    if not check_kaggle_credentials():
        raise RuntimeError("Configura las credenciales de Kaggle primero.")

    DOWNLOAD_DIR.mkdir(exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"[Kaggle] Descargando conjunto de datos: {KAGGLE_DATASET} ...")
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DOWNLOAD_DIR),
        unzip=True,
        force=force,
        quiet=False,
    )
    print(f"[Kaggle] ✅ Descarga completada en {DOWNLOAD_DIR}/")

    # Buscar archivos .csv dentro de la carpeta descargada
    csv_files = list(DOWNLOAD_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontró ningún archivo CSV en {DOWNLOAD_DIR}. "
            "Verifica que el dataset en Kaggle contenga archivos .csv."
        )

    # Preferir "TeamStatistics.csv" explícitamente
    team_csvs = [f for f in csv_files if "TeamStatistics.csv".lower() in f.name.lower()]
    chosen = team_csvs[0] if team_csvs else csv_files[0]
    print(f"[Kaggle] Usando el archivo: {chosen.name}")
    return chosen


def update_local_csv(new_csv_path: Path, force: bool = False):
    """
    Combina el CSV recién descargado con el CSV local existente.
    Elimina los duplicados basados en gameId y teamId para preservar el histórico de partidos.
    """
    new_df = pd.read_csv(new_csv_path, index_col=0, low_memory=False)
    print(f"[CSV] Nuevo conjunto de datos: {len(new_df):,} filas, {new_df.shape[1]} columnas")

    output_path = Path(OUTPUT_CSV)

    if output_path.exists() and not force:
        existing = pd.read_csv(output_path, index_col=0, low_memory=False)
        print(f"[CSV] Conjunto de datos existente: {len(existing):,} filas")

        # Combinar y eliminar duplicados inteligentemente
        combined = pd.concat([existing, new_df], ignore_index=True)

        # Identificar columnas de deduplicación
        dedup_cols = []
        if "gameId" in combined.columns and "teamId" in combined.columns:
            dedup_cols = ["gameId", "teamId"]
        elif "game_id" in combined.columns:
            dedup_cols = ["game_id"]

        if dedup_cols:
            before = len(combined)
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            after  = len(combined)
            print(f"[CSV] Deduplicación: {before - after} filas repetidas fueron removidas")
        else:
            combined = combined.drop_duplicates()

        print(f"[CSV] Total final: {len(combined):,} filas únicas listas para aprender")
    else:
        combined = new_df
        if force:
            print("[CSV] Modo forzado activado: sobreescribiendo el archivo local existente")

    # Ordenar cronológicamente si existe la fecha del partido
    if "gameDateTimeEst" in combined.columns:
        combined["gameDateTimeEst"] = pd.to_datetime(
            combined["gameDateTimeEst"], errors="coerce", utc=True
        )
        combined = combined.sort_values("gameDateTimeEst").reset_index(drop=True)

    combined.to_csv(output_path)
    print(f"[CSV] ✅ Archivo maestro guardado: {output_path} ({len(combined):,} filas)")
    return combined


def run(force: bool = False):
    """Flujo completo automatizado: Descargar → Unir → Guardar → Limpiar."""
    print(f"\n{'='*55}")
    print(f"  NBA Extractor Automático de Kaggle — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    try:
        csv_path = download_dataset(force=force)
        df = update_local_csv(csv_path, force=force)

        # Limpiar basura y temporales de la RAM y disco
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        print(f"\n✅ {OUTPUT_CSV} fue actualizado exitosamente.")
        print(f"   Filas Totales: {len(df):,}")
        print(f"   Rango del Histórico: {df['gameDateTimeEst'].min()} → {df['gameDateTimeEst'].max()}"
              if "gameDateTimeEst" in df.columns else "")

    except Exception as e:
        print(f"\n❌ Error Crítico: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Extractor de Datos Kaggle")
    parser.add_argument("--force", action="store_true",
                        help="Sobreescribe el archivo CSV local sin combinar los históricos")
    args = parser.parse_args()
    run(force=args.force)
