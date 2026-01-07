# Pok&#233;-ViT

This is the code I used to create a simple Vision Transformer (ViT) model to classify Pok&#233;mon. 

Tools used: PyTorch, CUDA, scikit-learn

## How to setup

If you would like to run the code locally, setup a venv with Python 3.11. Other Python versions may be possible, but you will have to modify the dependencies.

Once you've run `python3.11 -m venv .venv`, run `pip install -r requirements.txt` to install the appropriate dependencies. This project used PyTorch 2.9.1 with CUDA 13.0, if you wish to use different versions, you will have to modify the requirements.txt file.

## Data

For most testing, this project used an open-source dataset from kaggle: https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000

I also experimented with web-scraping to collect image data, using Serebii and Bulbapedia as sources, if you want to take a look at that code, it's in `scripts/scrape_pokemon.py`.

## Running Scripts

### `main.py`

| Option Syntax | Description |
| ------------- | ----------- |
| --data DATA    |       Path to dataset root (ImageFolder)|
|  --img-size IMG_SIZE |  Size of the images that the transformer will use |
|  --batch-size BATCH_SIZE|  |
|  --epochs EPOCHS| Number of epochs to train|
|  --lr LR| Learning Rate |
|  --weight-decay WEIGHT_DECAY||
|  --val-split VAL_SPLIT||
|  --seed SEED||
|  --num-workers NUM_WORKERS||
|  --out OUT        |     Path to save checkpoint
|  --pretrained     |     Use pretrained ViT from timm if available
|  --train-csv TRAIN_CSV|                        Optional train CSV (image_path,label) to load instead of ImageFolder
|  --val-csv VAL_CSV  |   Optional val CSV (image_path,label) to load instead of ImageFolder|
|  --dataset-root DATASET_ROOT |Root directory that image paths in CSV are relative to |

### `test.py`

| Option Syntax | Description |
| ------------- | ----------- |
|--model MODEL   |      Path to model checkpoint
  --data DATA     |      Path to dataset root (ImageFolder)
  --img-size IMG_SIZE
  --batch-size BATCH_SIZE
  --device DEVICE
  --out-csv OUT_CSV  |  Optional CSV to write predictions
  --csv CSV       |      Optional CSV file (image_path,label) to evaluate instead of ImageFolder
  --dataset-root DATASET_ROOT|  Root directory that image paths in CSV are relative to

### `create_splits_from_metadata.py`

| Option Syntax | Description |
| ------------- | ----------- |
| --metadata METADATA |  Path to metadata.csv
  --dataset-root DATASET_ROOT | Root that image paths in metadata are relative to
  --out-dir OUT_DIR  |   Directory where train/val/test dirs and CSVs will be created
  --train-ratio TRAIN_RATIO
  --val-ratio VAL_RATIO
  --test-ratio TEST_RATIO
  --seed SEED
  --copy   | Copy image files into train/val/test folders (default: only write CSVs)