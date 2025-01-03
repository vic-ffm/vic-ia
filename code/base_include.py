from pathlib import Path
from packaging import version
import datetime

# User settings
RANDOM_SEED = 20240124

# Main directory
DATA_DIRECTORY = Path('/Workspace').joinpath('Users', 'elena.tartaglia@deeca.vic.gov.au', 'vic-ia', 'data')

# Modelling dataset
MODEL_INPUT = DATA_DIRECTORY / 'ifrd_v2.5.8.csv'

# Figures for FAM paper
PAPER_FIGURES = DATA_DIRECTORY / 'fam_paper_figures'
DENSITY = PAPER_FIGURES / 'density'
RESIDUALS = PAPER_FIGURES / 'residuals'
DIAGNOSTICS = PAPER_FIGURES / 'diagnostics'