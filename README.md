# Taming the curse of dimensionality: CNN-based surrogates trained on multi-dimensional data
Repository for python code used in [Taming the curse of dimensionality: CNN-based surrogates trained on multi-dimensional data](submitted to NeurIPS 2024)

## Dependencies
This project runs in Python3.
Dependencies:
	• Python 3
	• NumPy
	• PyTorch
	• Torchvision
	• h5py

Specific version information can be found in requirements.txt.

To install:

```
  conda install -r requirements.txt
```

or

```
  pip3 install -r requirements.txt
```

## Folder structure
- 'src/': Contains all source code
- 'data/': Contains sample data
- 'model/': Contains a pretrained surrogate model which can be used for UQ

## Description of included items
- 'requirements.txt' - list of dependencies
- 'src/build_surrogate.py' - script to train surrogate model using 1D and 2D data
- 'src/UQ_experiment.py' - script to run UQ task
- 'src/dense_ed_multidimensional.py' - surrogate model architecture
- 'src/utils_surrogate.py' - supporting utility functions for surrogate model
- 'src/utils_UQ.py' - supporting utility functions for UQ task
- 'data/data.txt' - a .txt file containing a link to the anonymized data repository*
- 'models/pretrained_surrogate_example.pth' - an example pre-trained surrogate model for running UQ

## Setup
1. Install required libraries, as noted above: e.g. `pip3 install -r requirements.txt`
2. If running on a dataset not included in this repository:
	i. Update load_data() in build_surrogate.py as needed
	ii. Update n_HFS_runs_per_file and n_LFS_runs_per_file in main() in build_surrogate.py
    iii. Update filenames for data in main() in build_surrogate.py
	iv. Update "filenames" for data in UQ_experiment.py
3. Run build_surrogate.py as follows:
```
python3 build_surrogate.py --n_lfs=500 --n_hfs=20 --results_filename=./results/test.txt
```
This will save results in the format [n_lfs, n_hfs, rmse_best, freq] where 'freq' indicates the approach used to generate the low-fidelity data.
In this example, we are instructing python to train a surrogate with 500 1D runs, 20 2D runs, and save the results at ./results/test.txt.

4. Update MODEL_PATH, DATA_PATH, RESULTS_PATH in UQ_experiment.py
5. Run UQ_experiment.py as follows:
```
python3 UQ_experiment.py
```




* For the purposes of anonymization for submission of this manuscript, we elected to use an online data repository which supports anonymized sharing. The authors were unable to find a free data repository service which allows data collections exceeding 10GB. For this reason, we've included a sample of datasets but were not able to upload the full dataset used in the study. Upon acceptance of this manuscript, we will make public our Github repository which contains our full dataset. However, the sample available in the online repository is sufficient to reproduce some of the models in our paper.