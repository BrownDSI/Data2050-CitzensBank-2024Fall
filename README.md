# CitizensBank-Fraud-Signature-Detection
This is the project about fraud signature detection. In scanned documents there are signature blocks. The project is to use deep learning computer vision techniques to identify and extract signatures. 

## Repository Structure

```
.
|-- data/                 # Folder containing the five datasets
|   |-- BHSig260/                        # Contain both BHSig260 Dataset (Hindi) and BHSig260 Dataset (Bengali)
|   |-- Cedar/                           # English
|   |-- Hansig/                          # Chinese
|   |-- Real_Fake_Signature/             # Turkish
|   |-- Signature_Verification_Dataset/  # Dutch
|
|-- src/                  # Source code for the project
|   |-- preprocessing/
|   |-- resnet50/
|   |-- vgg16/
|   |-- efficientnet_b0/
|   |-- inceptionv3/
|   |-- xception/

```

In the `src/` directory, each folder contains its own README file with detailed information about the implementation. For more details, please refer to the README file in the corresponding folder.

---

## Setup Instructions

To set up the environment, simply install the provided `environment.yaml` file:
```bash
conda env create -f environment.yaml
conda activate <environment_name>
```

---

For more details, please refer to the project report.

