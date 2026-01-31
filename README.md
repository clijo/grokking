# grokking modular addition 

<div align="center">
  <img src="./assets/grokking-mod-add.png" alt="Train vs test accuracy">
  <br>
  <em>Fig. 1 - Train and test accuracy over epochs; each step is 100 epochs. Training reaches 100% quickly, while test accuracy stays near 0 until ~6000 epochs before rising to 100%.</em>
</div>


This project aims to reproduce "grokking" phenomenon in modular addition task, viz. $(a,b)\mapsto (a+b) \mod 113$, using the following 1-layer transformer model.

<div align="center">
  <img src="./assets/model.png" alt="Model architecture">
  <br>
  <em>Fig. 2 - Model architecture visualized using <a href="https://github.com/google-deepmind/treescope">Treescope</a>.</em>
</div>


Note that we do not use causal attention; it is not a `seq2seq` model.


# Reproduce 

1. Install `uv`.
2. Clone this repo and `cd` into it. Setup environment using `uv sync`.
3. (Optional) Setup `wandb` using `uv run wandb login`.
4. Check the available config options using `uv run run.py -h`.
4. Run the following command: 
    ```bash
    uv run run.py --p=113 --train_frac=0.20 --lr=1e-3 --epochs=30_000 --weight_decay=1.0
    ```

# Note 
- Fig 1 was created using by using 20% of the dataset as training set.
- `weight_decay` is a key parameter to induce grokking.