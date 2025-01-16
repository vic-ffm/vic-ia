from pathlib import Path
from pconfig import * # HOME_DIRECTORY is defined here by the user

# Name and location of the vic-ia repo
REPO_DIRECTORY = HOME_DIRECTORY / 'vic-ia'

# Main directory
DATA_DIRECTORY = REPO_DIRECTORY / 'data'
# DATA_DIRECTORY = Path('.').joinpath('data') # When called via reticulate, needs this file path.

# Modelling dataset
MODEL_INPUT = DATA_DIRECTORY / 'ifrd_v2.5.8.csv'

# Processed model input data for use in R 
PROCESSED_DATA = DATA_DIRECTORY / 'processed'
Path(PROCESSED_DATA).mkdir(parents=True, exist_ok=True)
## Output for GAMs
GRASS_GAM_RESULTS = PROCESSED_DATA / 'incidents_grass_test_gam_results.csv'
FOREST_GAM_RESULTS = PROCESSED_DATA / 'incidents_forest_test_gam_results.csv'

# Random seed used in modelling code
RANDOM_SEED = 20240124