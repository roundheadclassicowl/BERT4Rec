# Introduction

This repository is for our ESE546 Course Project. We attempt to improve BERT4Rec model from the following paper.

> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**

We incorparate the original BERT4Rec model with genre embeddings and user embedding respectivly. The model and performace is as follows.

The BERT4Rec model in this repo is based on the code from <a href="https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch"> BERT4Rec-VAE-Pytorch</a>. And the baseline model is base on <a href="https://github.com/recommenders-team/recommenders/tree/main/"> Recommenders</a>

# Models and Usage

## Overall

Run `main.py` with arguments to train and/or test you model. There are predefined templates for all models.

On running `main.py`, it asks you whether to train on MovieLens-1m or MovieLens-20m. (Enter 1 or 20)

After training, it also asks you whether to run test set evaluation on the trained model. (Enter y or n)

## BERT4Rec + Genre Embedding

### Usage

## BERT4Rec + User Embedding

### Usage

## Baseline

We use popularity-based baseline, which will recommend most-rated movie to users.

### Usage

Just go to the 'baseline' folder and run baseline.ipynb notebook.

## Examples

1. Train BERT4Rec on ML-20m and run test set inference after training

   ```bash
   printf '20\ny\n' | python main.py --template train_bert
   ```

# Test Set Results

We test our models on MovieLens-20m dataset.

<img src=Images/test_result.png>
