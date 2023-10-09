import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nltk
from collections import Counter
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn

FloatArray = NDArray[np.float64]

def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding

def logit(x: FloatArray) -> FloatArray:
    """Compute logit (inverse sigmoid)."""
    return np.log(x) - np.log(1 - x)

def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize vector so that it sums to 1."""
    return x / torch.sum(x)

class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()
        s0 = logit(np.ones((V, 1)) / V)
        self.s = nn.Parameter(torch.tensor(s0.astype("float32")))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        p = normalize(torch.sigmoid(self.s))
        return torch.sum(input, 1, keepdim=True).T @ torch.log(p)

def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", "NaN"]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize
    tokens = [char for char in text if char in vocabulary]

    # COMPUTE known_optimal_probs and known_minimum_loss
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    known_optimal_probs = [token_counts[token] / total_tokens for token in vocabulary]
    known_minimum_loss = -sum([token_counts[token]*np.log(known_optimal_probs[vocabulary.index(token)]) for token in token_counts])

    # generate one-hot encodings
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype("float32"))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations = 700
    learning_rate = 0.004

    # initialize an empty list to store losses
    losses = []

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_iterations):
        p_pred = model(x)
        loss = -p_pred
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # visualize the losses
    plt.figure(figsize=(10,6))
    plt.plot(losses, label='Training Loss')
    plt.axhline(y=known_minimum_loss, color='r', linestyle='--',label='Known Minimum Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Number of Iterations')
    plt.legend()
    plt.show()

    # Compute final token probabilities
    final_probs = torch.sigmoid(model.s).detach().numpy().flatten()

    # Visualize final token probabilities
    plt.figure(figsize=(10, 6))
    plt.bar(vocabulary, final_probs, label='Model Probabilities')
    plt.bar(vocabulary, known_optimal_probs, alpha=0.5, label='Known Optimal Probabilities')
    plt.xlabel('Tokens')
    plt.ylabel('Probability')
    plt.title('Token Probabilities vs. Optimal Probabilities')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    gradient_descent_example()
