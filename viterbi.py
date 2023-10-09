"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Patrick Wang, 2021
"""
from typing import Sequence, Tuple, TypeVar
import numpy as np
import nltk
from collections import defaultdict, Counter

Q = TypeVar("Q")
V = TypeVar("V")

def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)


# 1. Data preparation

nltk.download('brown')
nltk.download('universal_tagset')
from nltk.corpus import brown

tagged_sents = brown.tagged_sents(tagset='universal')[:10000]

tags = set()
words = set()
for sent in tagged_sents:
    for word, tag in sent:
        tags.add(tag)
        words.add(word.lower())

words.add("OOV/UNK")

tag2idx = {tag: idx for idx, tag in enumerate(tags)}
word2idx = {word: idx for idx, word in enumerate(words)}


# 2. HMM components generation
pi = np.zeros(len(tags))
for _, tag in tagged_sents[0]:
    pi[tag2idx[tag]] += 1
pi = pi / np.sum(pi)

A = np.zeros((len(tags), len(tags)))
for sent in tagged_sents:
    for i in range(len(sent) - 1):
        A[tag2idx[sent[i][1]], tag2idx[sent[i+1][1]]] += 1

A += 1
A = A / A.sum(axis=1, keepdims=True)

B = np.zeros((len(tags), len(words)))
for sent in tagged_sents:
    for word, tag in sent:
        B[tag2idx[tag], word2idx[word.lower()]] += 1

for tag in tags:
    B[tag2idx[tag], word2idx["OOV/UNK"]] = 1

B += 1
B = B / B.sum(axis=1, keepdims=True)


# 3. Inference with Viterbi

test_sents = brown.tagged_sents(tagset='universal')[10150:10153]

for sent in test_sents:
    obs = [word2idx.get(word.lower(), word2idx["OOV/UNK"]) for word, _ in sent]
    states, _ = viterbi(obs, pi, A, B)
    predicted_tags = [list(tag2idx.keys())[state] for state in states]

    true_tags = [tag for _, tag in sent]
    print("True:", true_tags)
    print("Pred:", predicted_tags)
    print("\n")

