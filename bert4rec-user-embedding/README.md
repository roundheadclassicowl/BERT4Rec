# User Embedding Contribution

This repo is developed from [a thorough reproduction](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) of [the original BERT4Rec paper](https://arxiv.org/pdf/1904.06690.pdf) specifically on the user embedding component. The key changes are listed below in this [commit](https://github.com/roundheadclassicowl/BERT4Rec/commit/8b8f31b1c41dbc2551bc392eb72ab7a81f62d916).
## How to run

1. Run the following terminal command to start preprocessing the 20M dataset. Terminate the process once it is done.

   ```bash
   printf '20\ny\n' | python main.py --template train_bert
   ```

2. Edit two hard-coded path in `models/bert_modules/embedding/genre.py` to find the inverse smap and and `movies.csv`.

3. Rerun the training code above.

# Introduction (content below is from original repo)

This repository implements models from the following two papers:

> **BERT4Rec: Sequential Recommendation with BERT (Sun et al.)**  

> **Variational Autoencoders for Collaborative Filtering (Liang et al.)**  

and lets you train them on MovieLens-1m and MovieLens-20m.

# Usage

## Overall

Run `main.py` with arguments to train and/or test you model. There are predefined templates for all models.

On running `main.py`, it asks you whether to train on MovieLens-1m or MovieLens-20m. (Enter 1 or 20)

After training, it also asks you whether to run test set evaluation on the trained model. (Enter y or n)

## BERT4Rec

```bash
python main.py --template train_bert
```

## DAE

```bash
python main.py --template train_dae
```

## VAE

### Search for the optimal beta

```bash
python main.py --template train_vae_search_beta
```

### Use the found optimal beta

First, **fill out the optimal beta value in `templates.py`**. Then, run the following.

``` bash
python main.py --template train_vae_give_beta
```

<img src=Images/vae_tensorboard.png width=800>

The `Best_beta` plot will help you determine the optimal beta value. It can be seen that the optimal beta value is 0.285.

The gray graph in the `Beta` plot was trained by fixing the beta value to 0.285.

The `NDCG_10` metric shows that the improvement claimed by the paper has been reproduced.

## Examples

1. Train BERT4Rec on ML-20m and run test set inference after training

   ```bash
   printf '20\ny\n' | python main.py --template train_bert
   ```

2. Search for optimal beta for VAE on ML-1m and do not run test set inference

   ```bash
   printf '1\nn\n' | python main.py --template train_vae_search_beta
   ```
  
# Test Set Results

Numbers under model names indicate the number of hidden layers.

## MovieLens-1m

<img src=Images/ML1m-results.png>

## MovieLens-20m

<img src=Images/ML20m-results.png>
