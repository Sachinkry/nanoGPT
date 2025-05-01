# nanoGPT - Minimal Character-Level GPT from Scratch

This project is a stripped-down implementation of a character-level GPT (Generative Pretrained Transformer) written in PyTorch. It's based on Karpathy's teaching code, progressively evolved to support features like self-attention, multi-head transformers, and basic training/evaluation loops.

## 🧠 What's Inside

### Models
- `bigram.py`: Simplest possible language model using bigram probabilities.
- `v2.py`, `v3.py`: Iterative improvements, with increasing model depth and complexity—transformer blocks, layer norm, dropout, etc.

### Training
- `train_gpt2.py`: Main training script for GPT models.
- `metrics.py`: Evaluation utilities.
- `metrics_logs.txt` / `eval_logs.txt`: Training and evaluation logs (hint: these are your early diagnostics and debugging helpers).

### Data
- `input.txt`: Shakespeare sample input (used for training).
- `play.ipynb`: Jupyter notebook for model experimentation and quick runs.

## ⚙️ How to Run It

```bash
# Download tiny shakespeare corpus
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Run the bigram model
python bigram.py

# Or run the training script (after adjusting hyperparams if needed)
python train_gpt2.py
