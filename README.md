
---

# Tweet Classification using ALBERT + Data Augmentation

This project fine-tunes ALBERT (or TinyBERT) to classify tweets as either *INFORMATIVE* or *UNINFORMATIVE*. The pipeline includes LLM-based data augmentation, progressive unfreezing, mixed training with hard examples, and evaluation utilities. The dataset is derived from the **WNUT-2020 Task 2** competition.

---

##  File Overview

| File                      | Description                                                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `augment_dataframe.py`    | Defines the LLM-based tweet augmentation logic with multi-threading support using OpenAI-compatible clients.                                  |
| `batch_train.py`          | Main training pipeline: includes dataloader setup, optimizer, training loop, validation, and model checkpointing with progressive unfreezing. |
| `DownloadNecessaries.py`  | Downloads and saves TinyBERT and ALBERT base models and tokenizers to `model_assets/`.                                                        |
| `filter_misclassified.py` | Filters out misclassified samples from an augmented dataset to generate a hard example set for fine-tuning.                                   |
| `finetune.py`             | Performs a second stage of training using a mix of original and hard samples with flexible ratio control.                                     |
| `generate_csv.py`         | Runs inference on the test dataset and produces a submission-ready `prediction.csv`.                                                          |
| `ProjectDataloader.py`    | Custom `TweetDataset` class for encoding and preprocessing tweet data into PyTorch-compatible tensors.                                        |
| `run_augmentation.py`     | Script that loads the original training data and augments each sample using `augment_dataframe.py`.                                           |
| `test.py`                 | Tests a trained model on the labeled test set and reports classification accuracy.                                                            |
| `unfreezer.py`            | Implements progressive unfreezing of transformer layers either by epoch or by validation accuracy.                                            |

---

##  Folder Structure

```bash
.
├── data/
│   ├── WNUT-2020-Task-2-Dataset/        # Provided dataset (train/valid/test .tsv files)
│   ├── augment_data/                    # Augmented tweets generated via LLM
│   └── misclassified_augment_data/      # Hard samples misclassified by model, used for fine-tuning
│
├── model_assets/
│   ├── tinybert/                        # Pretrained TinyBERT (downloaded)
│   ├── albert/                          # Pretrained ALBERT (downloaded)
│   ├── albert_batch_best/              # Best checkpoints from `batch_train.py` (ALBERT)
│   ├── albert_batch_training/          # Intermediate checkpoints from `batch_train.py`
│   └── Albert_Finetune/                # Fine-tuned checkpoints trained on hard + original samples
│
├── *.py                                 # Main scripts (see File Overview)
└── README.md                            # This file
```


## Reproduction Instructions

To reproduce the complete training pipeline, follow the steps below. Steps marked as *(No need to run)* are already completed, and the output files are included in this repository.

### Step 0: Preparation *(No need to run)*

The script `DownloadNecessaries.py` downloads and saves the base models **ALBERT** and **TinyBERT**. While both are supported, only ALBERT was used in the final experiments due to better performance.
**Note:** You do **not** need to run this script — the pre-downloaded models are already included in the `model_assets/` folder.

---

### Step 1: Initial Training

Run:

```bash
python batch_train.py
```

* This performs training using the original training set.
* All random seeds are fixed for reproducibility.
* With the given hyperparameters, the model achieves a **best validation accuracy of 0.90** at **epoch 11**.
* Early stopping is triggered shortly after due to no further improvement.

---

### Step 2: Data Augmentation *(No need to run)*

The script `run_augmentation.py` uses the **Qwen** LLM API to augment the original training data.

* This step requires a valid API key.
* Due to the stochastic nature of large language models, the output may differ slightly on each run.
* The pre-generated augmented dataset is already included at:

```bash
./data/augment_data/augmented_train_dataset.tsv
```

You **do not need to run this step** again.

---

### Step 3: Misclassified Sample Mining *(No need to run)*

The script `filter_misclassified.py` uses the best-performing model from Step 1 to evaluate the augmented dataset and extract **misclassified samples** (i.e., hard examples).

* These samples are saved to:

```bash
./data/misclassified_augment_data/misclassified_augmented_data.tsv
```

You **do not need to rerun this step** — the file is already provided.

---

### Step 4: Fine-Tuning on Mixed Dataset

Run:

```bash
python finetune.py
```

* This retrains the model using a **1:1 mixture** of the original training set and the hard sample set.
* The learning rate is reduced to better adapt the pretrained model.
* The best model is obtained at **epoch 2**, achieving **test accuracy of 89.1%**.

 **Note:** This step builds on the best model saved in Step 1.

If the script reports that the pretrained model cannot be found, **please modify line 241** of `finetune.py` and set the correct model checkpoint path (e.g., `model_epoch_11_valacc_0.9000.pt` under `model_assets/albert_batch_best/`).

---

### Step 5: Testing and CSV Generation

To evaluate the final model and prepare a submission file for platforms like Kaggle:

```bash
python test.py
python generate_csv.py
```

This will generate a `prediction.csv` file in the project root directory.



---



