# breast-mri-fgt-segmentation

This project focuses on the segmentation of **Fibroglandular Tissue (FGT)** in
breast MRI images using deep learning.

## Dataset

The Duke Breast Cancer MRI dataset consists of breast MRI scans stored in
**DICOM format**.

Segmentation masks are provided as DICOM-SEG objects. Each segmentation file
contains multiple labeled segments, which are mapped to the corresponding MRI
images using patient and series metadata. In this project, the segments
corresponding to **Mammary Fibroglandular Tissue (FGT)** are extracted and
combined to form a binary ground-truth mask.

The dataset files are downloaded but cannot be directly read without proper
retrieval and organization. Therefore, the data must be downloaded using
**NBIA Data Retriever (version 4.4)**.

Due to dataset size constraints, the data itself is **not included**
in this repository.

## Notes on Dataset Size and Code Usage

If you want to test the code using only a small subset of the dataset,
please keep the **commented-out evaluation blocks** disabled.

In this setup, model behavior can be assessed using training loss,
Dice score, IoU, and the number of predicted positive pixels versus
ground-truth positive pixels.

If a larger portion of the dataset (e.g., at least half of the cases)
is available, the **commented evaluation sections** can be enabled to
perform full training, validation, and testing, with corresponding loss and evaluation metrics.

## Method
- 2D U-Net architecture
- Binary segmentation (FGT vs background)
- Loss function: Binary Cross-Entropy (BCE) + Dice Loss

## Evaluation
The following metrics were used for evaluation:
- Training loss
- Validation loss
- Test loss
- Dice score
- Intersection over Union (IoU)
- Precision

## Notes
This repository contains only the code required to reproduce the experiments.
