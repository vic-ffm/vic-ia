from pathlib import Path
from packaging import version
import datetime

# User settings
RANDOM_SEED = 20240124

# Main directory
DATA_DIRECTORY = Path('/Workspace').joinpath('Users', 'elena.tartaglia@deeca.vic.gov.au', 'vic-ia', 'data')

# Modelling dataset
MODEL_INPUT = DATA_DIRECTORY / 'ifrd_v2.5.8.csv'

# Processed model input/output for GAMs R code
PROCESSED_DATA = DATA_DIRECTORY / 'processed'
Path(PROCESSED_DATA).mkdir(parents=True, exist_ok=True)
GRASS_MODEL_INPUT_CSV = DATA_DIRECTORY / PROCESSED_DATA / 'incidents_modelling_grass.csv'
FOREST_MODEL_INPUT_CSV = DATA_DIRECTORY / PROCESSED_DATA / 'incidents_modelling_forest.csv'
GRASS_GAM_RESULTS = DATA_DIRECTORY / PROCESSED_DATA / 'incidents_grass_test_gam_results.csv'
FOREST_GAM_RESULTS = DATA_DIRECTORY / PROCESSED_DATA / 'incidents_forest_test_gam_results.csv'

