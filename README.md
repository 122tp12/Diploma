# Classification of Handwritten Strokes using Graph Neural Networks

## Overview
This repository provides a Graph Neural Network (GNN) implementation for the automated classification of handwritten strokes using the IAMonDo-db dataset. By representing digital ink documents as relational graphs—where nodes correspond to individual strokes and edges represent spatio-temporal relationships—the model effectively classifies elements into binary (Text vs. Non-Text) or multiple categories (Text, Table, Formula, Diagram, Other).

## Model Architecture
The core model is a modified Edge-Conditioned Graph Attention Network (EGAT) designed to handle deep architectures without over-smoothing.
* **Node Updates:** Utilizes `GATv2Conv` for dynamic multi-head attention.
* **Edge Updates:** Replaces standard linear transformations with a Gated Recurrent Unit (`GRUCell`) to treat edge features as hidden memory states.
* **Virtual Node:** Incorporates a virtual node connected to all real nodes to capture global document context, filtering information through the GRU mechanism to reduce noise.

## Dataset and Features
The project utilizes the **IAMonDo-db** dataset, storing digital handwritten documents in InkML format. The `features.py` script computes a comprehensive set of geometric and kinematic properties prior to training:
* **Node Features (23 total):** Includes trajectory length, convex hull area, temporal duration, principal axis ratios, and spatial/temporal neighbor statistics.
* **Edge Features (19 total):** Captures relative relationships such as minimum distance between strokes, bounding box ratios, off-stroke distances, and temporal gaps.

## Project Structure
* `IAMonDo-db-1.0/`: Directory containing the dataset repository.
* `EGAT.py`: Contains the PyTorch definitions for the `EGATLayer` and `EGAT_model`, along with training and validation step functions.
* `features.py`: Logic for extracting the 42 spatial and temporal features from the raw stroke data.
* `main.py`: The primary execution script for data splitting, training loops, validation, testing, and generating plots for misclassified cases.
* `batchering.py`: Scripts for parsing InkML files, handling complex cases, and applying augmentations like rotation and scaling.
* `StrokeGraphDataset_class.py`: PyTorch Dataset implementation for loading pre-processed `.pt` graph data batches.
* `ploting.py`: Scripts for visualizing training trajectories and comparing metrics.
* `metrics.ipynb` & `results.ipynb`: Jupyter notebooks for evaluating metrics and deeply analyzing classification results.

## Training and Optimization
The training pipeline in `main.py` is optimized for handling imbalanced graph data and complex topologies.
* **Optimization:** Uses the `AdamW` optimizer combined with a `ReduceLROnPlateau` learning rate scheduler and Gradient Norm Clipping (max norm of 1.0).
* **Class Weighting:** Applies explicit class weights within the Cross-Entropy loss function to penalize errors in minority classes (e.g., Tables, Formulas).
* **Data Augmentation:** Generates synthetic training batches by applying random rotations (-15° to 15°) and scaling factors (0.8 to 1.2).
* **Early Stopping:** Prevents overfitting by halting training when the validation loss plateaus.

## Evaluation & Visualization
* **Metrics:** Evaluates model performance using `sklearn.metrics` to track Accuracy, Balanced Accuracy, Matthews Correlation Coefficient (MCC), Jaccard Score, and Log Loss.
* **Complexity:** Calculates computational cost (MegaFLOPs) using the `fvcore` library.
* **Error Analysis:** Includes a plotting utility that visually flags misclassified strokes, coloring false negatives and false positives (e.g., cyan, gold, gray) to aid in qualitative analysis and troubleshooting.
