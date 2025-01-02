from pathlib import Path
from packaging import version
import datetime

# User settings
current_join_version = "2.5.8"
current_model_version = "2.5.8"
RANDOM_SEED = 20240124

# Main directory
DATA_DIRECTORY = Path('/dbfs').joinpath('mnt', 'raw', 'suppression')

# Source folders
INCIDENTS_FOLDER = DATA_DIRECTORY / 'incidents_clean'
JOIN_DATA_FOLDER = DATA_DIRECTORY / 'join_data'
FIRE_WEATHER_DISTRICTS_FOLDER = JOIN_DATA_FOLDER / 'fire_weather_districts'
SOIL_MOISTURE_FOLDER = JOIN_DATA_FOLDER / 'soil_moisture' / 'historical_absolute'
FFMV_DISTRICTS_FOLDER = JOIN_DATA_FOLDER / 'ffmv_fire_districts'

# Incidents datset
INCIDENTS_VERSION = f'release_version_{current_join_version}'
INCIDENTS_INPUT = INCIDENTS_FOLDER / INCIDENTS_VERSION / 'incidents.pkl'
INCIDENTS_OUTPUT = INCIDENTS_FOLDER / INCIDENTS_VERSION / 'incidents_join.pkl'

# Modelling dataset
MODEL_INPUT_VERSION = f'release_version_{current_model_version}'
MODEL_INPUT = INCIDENTS_FOLDER / MODEL_INPUT_VERSION / 'incidents_join.pkl'

# Check latest release version
version_folders = INCIDENTS_FOLDER.glob("release_version_*")
version_numbers = [folder.name.split("_")[-1] for folder in version_folders]
latest_version_number = max(version.parse(n) for n in version_numbers)

LATEST_VERSION = f'release_version_{latest_version_number}'

# TODO: MODEL_INPUT doesn't necessarily exist (e.g. when calling this from create_dataset.py)
# Print version info
# creation_times = [datetime.datetime.fromtimestamp(p.stat().st_ctime).strftime('%Y-%m-%d, %H:%M:%S') for p in [
#     INCIDENTS_FOLDER / LATEST_VERSION / 'incidents.pkl', INCIDENTS_INPUT, MODEL_INPUT
# ]]
# print("Latest source version available:", LATEST_VERSION, f"({creation_times[0]})")
# print("Current version for feature generation:", INCIDENTS_VERSION, f"({creation_times[1]})")
# print("Current version for model input:", MODEL_INPUT_VERSION, f"({creation_times[2]})")

# Datasets to be joined
WEATHER_PRED = Path('/dbfs').joinpath('mnt', 'raw') / 'weather' / 'adfd-processed'
WEATHER_HIST = Path('/dbfs').joinpath('mnt', 'raw') / 'weather' / 'vicclim-processed'
DEM_GRID = JOIN_DATA_FOLDER / 'DEM.tif'
BUILDING_POLYGONS = JOIN_DATA_FOLDER / 'VMFEAT_BUILDING_POLYGON.geojson'
ROADS = JOIN_DATA_FOLDER / 'VMTRANS/TR_ROAD_ALL.shp'
FUEL_TYPE_DATA = JOIN_DATA_FOLDER / 'fuel-type-data.txt'
FUEL_GRID = JOIN_DATA_FOLDER / 'fuel2021.tif'
FWD_NAME = 'IDM00007'
FIRE_WEATHER_DISTRICTS_FOLDER = JOIN_DATA_FOLDER / 'fire_weather_districts'
FFMV_FD_NAME = 'LF_DISTRICT'


# Generated datasets
RUGGEDNESS_GRID = JOIN_DATA_FOLDER / 'ruggedness.tif'
#ROAD_PROXIMITY_GRID = JOIN_DATA_FOLDER / 'generated_road_proximity.tif'
BUSH_GRASS_DISTANCE_GRID = JOIN_DATA_FOLDER / 'bush_grass_distance.tif'

# Figures for FAM paper
PAPER_FIGURES = DATA_DIRECTORY / 'fam_paper_figures'
DENSITY = PAPER_FIGURES / 'density'
RESIDUALS = PAPER_FIGURES / 'residuals'
DIAGNOSTICS = PAPER_FIGURES / 'diagnostics'

