This folder contains the code for the benchmark of the imputers on the OpenML datasets.

The code is organized as follows:

- `create_openml_missingness_datasets.py`: This script is used to create the missingness datasets from the OpenML datasets.
- `get_openml_errors.py`: This script is used to get the errors of the imputers on the OpenML datasets.
- `plot_error_violinplots.py`: This script is used to plot the violin plots of the errors of the imputers on the OpenML datasets.
- `plot_error_boxplots.py`: This script is used to plot the box plots of the errors of the imputers on the OpenML datasets.
- `plot_negative_rmse.py`: This script is used to plot the negative RMSE of the imputers on the OpenML datasets.

The datasets are stored in the `datasets` folder. The figures are stored in the `figures` folder.

## UCI Datasets

We test on the same datasets as in the HyperImpute paper:

- Airfoil Self-Noise
- Blood Transfusion
- California Housing
- Concrete Compression
- Diabetes
- Ionosphere
- Iris
- Letter Recognition
- Libras Movement
- Spam Base
- Wine Quality (Red)
- Wine Quality (White)