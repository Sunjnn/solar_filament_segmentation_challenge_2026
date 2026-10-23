# %% md
# # Cross-Site & Temporal Generalization Experiments
#
# Diagnose how well models generalize across:
# - Different observatories (KSO, Meudon, BBSO, Uccle)
# - Different time periods
# - Different solar activity levels
#
# This is analogous to the cross-domain experiments in the flight satisfaction project.

# %% md
# ## 1. Setup

# %%
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

# %% md
# ## 2. Experiment Design
#
# ### Leave-One-Site-Out (LOSO)
# Train on 3 observatories, test on the held-out one.
# This reveals whether models memorize site-specific artifacts.
#
# ### Temporal Split
# Train on early dates, test on later dates.
# This reveals whether models degrade over time (e.g., due to solar cycle changes).
#
# ### Metrics
# - PQ per site
# - PQ per time period
# - False positive rate per site
# - False negative rate per site

# %% md
# ## 3. Data Splitting

# %%
# TODO: Create site-based splits
# TODO: Create temporal splits
# TODO: Verify no leakage

# %% md
# ## 4. Run Experiments

# %%
# TODO: Train model on each split
# TODO: Evaluate on held-out data
# TODO: Collect per-site + per-period metrics

# %% md
# ## 5. Results & Visualization

# %%
# TODO: Bar chart of PQ per observatory
# TODO: Line chart of PQ over time
# TODO: Confusion matrix per site (predicted vs actual filaments)
# TODO: Statistical significance tests

# %% md
# ## 6. Conclusions

# %%
# TODO: Which sites are hardest?
# TODO: Does temporal drift exist?
# TODO: Recommendations for robust models