# Fitness Tracker — Barbell Exercise Classification & Rep Counting

A machine learning project that processes wristband sensor data (accelerometer + gyroscope) to **classify barbell exercises** and **count repetitions** automatically. The goal is to mimic the kind of automatic exercise tracking that smartwatches do, but built from scratch on raw IMU data.

## What it does

Given raw motion data recorded from a [MbientLab MetaMotion](https://mbientlab.com/) sensor worn on the wrist, the pipeline:

1. **Loads and cleans** raw CSV files from accelerometer (12.5 Hz) and gyroscope (25 Hz) sensors.
2. **Resamples** both streams to a common 200 ms timestep and merges them into one dataset.
3. **Removes outliers** using Chauvenet's criterion (after comparing it against IQR and Local Outlier Factor).
4. **Engineers features**: Butterworth low-pass filtering, Principal Component Analysis, sum-of-squares magnitudes, temporal abstraction (rolling mean/std), Fourier-based frequency features, and K-means clustering.
5. **Trains and compares classifiers** — Neural Network, Random Forest, KNN, Decision Tree, and Naive Bayes — across multiple feature subsets, using forward feature selection and grid search.
6. **Counts repetitions** with a peak-detection algorithm on the low-pass-filtered signal.

## Exercises classified

- Bench Press
- Squat
- Overhead Press (OHP)
- Deadlift
- Row
- Rest

## Project structure

```
├── data/
│   ├── raw/MetaMotion/         <- Raw sensor CSVs
│   ├── interim/                <- Intermediate pickled datasets
│   └── processed/              <- Final datasets ready for modeling
├── models/                     <- Trained model artifacts
├── notebooks/                  <- Exploratory notebooks
├── reports/figures/            <- Generated plots (per exercise / participant)
├── src/
│   ├── data/make_dataset.py            <- Build the merged & resampled dataset
│   ├── features/
│   │   ├── remove_outliers.py          <- Chauvenet / IQR / LOF outlier handling
│   │   ├── build_features.py           <- Filtering, PCA, temporal & frequency features, clustering
│   │   ├── count_repetitions.py        <- Peak-detection rep counter
│   │   ├── DataTransformation.py       <- LowPassFilter & PCA helpers
│   │   ├── TemporalAbstraction.py      <- Rolling-window numerical features
│   │   └── FrequencyAbstraction.py     <- Fourier transform features
│   ├── models/
│   │   ├── LearningAlgorithms.py       <- Classifier wrappers (NN, RF, KNN, DT, NB)
│   │   └── train_model.py              <- Training, feature selection, evaluation
│   └── visualization/visualize.py      <- Plotting utilities
├── environment.yml             <- Conda environment definition
└── requirements.txt
```

## Setup

The project uses a conda environment.

```bash
conda env create -f environment.yml
conda activate tracking-barbell-exercises
```

Main dependencies: `python 3.11`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`.

## How to run

The scripts are designed to be run in order — each step writes a pickle that the next step reads from `data/interim/`.

```bash
cd src/data        && python make_dataset.py            # -> 01_data_processed.pkl
cd ../features     && python remove_outliers.py         # -> 02_outlies_removed_chauvenets.pkl
                     python build_features.py           # -> 03_data_features.pkl
cd ../models       && python train_model.py             # train + evaluate classifiers
cd ../features     && python count_repetitions.py       # rep counting + MAE benchmark
```

## Results

- The Random Forest classifier on the full feature set reaches very high accuracy on the held-out test split.
- A participant-out evaluation (training on 4 participants, testing on the 5th) is also included to check generalization.
- The rep counter is benchmarked with mean absolute error against the known rep counts (5 for heavy sets, 10 for medium sets).

Plots for each exercise/participant combination are saved under [reports/figures/](reports/figures/).

## Credits

This project is built by following the excellent **"Full Machine Learning Project"** tutorial series by **Dave Ebbelaar** — huge thanks to him for the clear, practical walkthrough of every step from raw sensor data to a working classifier.

- YouTube channel: [@daveebbelaar](https://www.youtube.com/@daveebbelaar)

The project layout is based on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template.

The Chauvenet's criterion and rolling-window abstraction code is adapted from [Mark Hoogendoorn & Burkhardt Funk — Machine Learning for the Quantified Self (ML4QS)](https://github.com/mhoogen/ML4QS).
