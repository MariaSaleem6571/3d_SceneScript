# 3D SceneScript

This repository contains the code for 3D Scene Reconstruction using Point Cloud Data, which predicts structured commands to recreate 3D scenes from point cloud inputs.

## Overview

The model developed in this project uses neural networks to convert point cloud data into structured language commands (e.g., `make_wall`, `make_door`, `make_window`) for architectural environments. The model architecture is inspired by Meta's **SceneScript**, and it has been trained on the **Aria Synthetic Environments (ASE)** dataset, which was also provided by Meta.

![Model Architecture](/images/model_architechture.png)
*Figure 1: Model architecture used for 3D scene reconstruction, inspired by Meta's SceneScript.*

The ASE dataset consists of procedurally generated scenes with point cloud data, making it ideal for testing and training models for architectural and indoor environment reconstruction.

### Installation
1. Clone the repository:
   ```bash

   git clone https://github.com/MariaSaleem6571/3d_SceneScript.git

2. Navigate to the project directory:
   ```bash
   cd 3d_SceneScript
   
3. Create a virtual environment:
   ```bash
   conda create -n 3dscenescript python=3.12
   conda activate 3dscenescript

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
## Training

The training of the model can be done using the `experiments.py` script, where the hyperparameters can be configured in the `experiment_config.py` file.

To start training, simply run:

```bash
python experiments.py
```

## Testing

For testing the model, use the `testing.py` script.

To run the testing, execute the following command:

```bash
python testing.py
```


