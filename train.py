import numpy as np
from model import build_seq2seq_attention
from data_preprocessing import load_and_preprocess

def train():
    # placeholder dummy arrays
    X = np.random.rand(100, 20, 5)
    y = np.random.rand(100, 20, 1)

    model = build_seq2seq_attention(input_dim=5, timesteps=20, output_dim=1)
    model.fit([X, X], y, epochs=3, batch_size=8)
    model.save("model.h5")

if __name__ == "__main__":
    train()
