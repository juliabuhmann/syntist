# SYNTIST

Pronounced *scientist*. A project for the automated detection of synaptic partners in Electron Microscopy brain data.

This is the original implementation of the paper ["Synaptic partner prediction from point annotations in insect brains"](https://arxiv.org/pdf/1806.08205.pdf).

## Status
Preliminary, not well documented version.  At the moment, this repos mainly serves as a reference for the paper to see the original U-Net architecture in `tensorflow` [here](tf_utils/unet_tinder.py) and the [training pipeline](train_on_cremi.py) in [`gunpowder`](https://github.com/funkey/gunpowder/tree/release-v1.0/gunpowder).


## Installation

No installation is required, but following dependencies:
- The gunpowder training pipeline currently runs with this hacky (old) gunpowder version:
  - https://github.com/Rekrau/gunpowder_syntist/tree/master_tinder_network
- tensorflow = 1.3.0
- [augment](https://github.com/funkey/augment)
- h5py

Note: We internally use docker, so this has not fully been tested.

### Using Docker
Sorry :( not yet available, but coming soon.

## Running the pipeline

### Training
Start the [training script](train_on_cremi.py)

### Inference
Inference scripts and trained networks coming soon. 
   
### Synaptic partner extraction
We provide a small test dataset cropped from CREMI dataset `Sample C`, which includes 1) our network predictions, 2) ground truth segmentation, 3) raw data that can be used to run the synaptic partner extraction script.

Donwload the dataset from [here](https://drive.google.com/open?id=1q-w5Y9ekt4EVHWtgE4-xTmRXEULXgDFc). You can have a look at the data with [this notebook](notebooks/visualize_data_with_nyroglancer.ipynb).

In order to obtain synaptic partners from the U-Net output and validate with CREMI stats, run this [extraction and evaluation script](extract_syn_partners.py).  As a record, we created the cropped dataset with [this script](data/prepare_small_crop.py).
