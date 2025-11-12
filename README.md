# VLM-Firewatch

This project proposes the development of a real-time wildfire detection system deployed on a drone equipped with a Jetson Orin Nano. We aim to begin classification of wildfires using a Visual Language Model (VLM). Our first step to achieving this goal is to begin searching for relevant datasets online. Through detailed research, we have decided to proceed with the FLAME dataset and datasets based on it (FLAME3, FlameVision) as well as the Places365 validation set (to prevent class imbalance). These datasets take advantage of sensor fusion and utilize it to combine the IR and feed data. Our first step is to aggregate these 3 datasets together into VQA format (question-answer, as shown below) such that our eventual model can classify images into fire being present and no fire being present.

## 0 - Creating a `conda` environment

We use `conda` to handle all needed Python dependencies. To work with a conda environment, you must first install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). Once installed, run the below to create the environment:

```console
conda env create -f env/environment.yml
```

Once the environment has been created, you must activate it by running:

```console
conda activate vlm_firewatch_env
```

## 1 - Setting up Source Datasets

Any code involving data for the project can be found in `src/data/`. This folder is responsible for loading, aggregating, and therefore processing all four of our source datasets into a unified dataset. We use the following four dataset linked below:
- FLAME
- FLAME3
- FlameVision
- Places365 (You do not need to download this manually.)

Once you have downloaded the datasets, you must extract them to a convenient location on your machine. Once extracted, navigate to `src/data/dataset_configs.py` and replace the below source paths with your paths. Note that the folders pointed to by the paths must match, so take great care to ensure that they do. You must also create a folder to store the unified dataset that we will create.

```python
flame_config = {
    "src": "<path-to>/FLAME/Training",
}

flame3_config = {
    "src": "<path-to>/flame3/FLAME 3 CV Dataset (Sycan Marsh)",
}

flamevision_config = {
    "src": "<path-to>/flamevision/Classification",
}

places_365_config = {
    "src": "<path-to>/places365",
    "processed": "<path-to>/places365/processed_nofire_samples"
}

unified_config = {
    "src": "<path-to>/unified_dataset"
}
```

> [NOTE] For the Places365 dataset, our code will automatically download it for you as long as the src path is set to a valid folder. We recommend that you create a folder named `places365` and set the source path to that. Remember to include that path for the processed path as well.

Next, we must create the unified dataset. To do this, you must run the `src/data/process_datasets.py` script. This script uses the dataset configuration to load, aggregrate, and process the unified dataset from the constituent datasets. To run the script, you must run:

```console
python src/data/process_datasets.py
```

## 2 - Training Baseline Classification Models and VLM

All of our training code can be found in `src/train/`. This directory includes configurations for fine-tuning, training (aka fine-tuning) scripts, and dataloaders to help load the unified dataset. The results from finetuning the baseline models will be stored in  `src/train/results` while the model weights themselves will be saved in `src/train/models`. We will discuss how the VLM is saved in that section later.

To train/fine-tune the classifier models, run the below:

```python
python train_efficientnet.py

python train_yolo.py
```

## 3 - Quantize the Fine-Tuned VLM

TODO for last checkpoint.
