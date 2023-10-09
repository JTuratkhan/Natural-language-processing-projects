import random
from collections import Counter
import nltk
import numpy as np

#считывание данных и разделение слов по токенам
corpus = nltk.word_tokenize(
        nltk.corpus.gutenberg.raw("austen-sense.txt").lower()
    )


#составление списка слов по порядку в зависимости от задаваемого параметра n
def build_ngram_model(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return ngrams


#ключи с со значениями, например рядом с love будет to, если мы выберем  n-2
def model_generation(corpus, n):
    model = {}
    for i in range(1, n + 1):
        ngrams = build_ngram_model(corpus, i)
        for ngram in ngrams:
            prefix, suffix = ngram[:-1], ngram[-1]
            if prefix not in model:
                model[prefix] = []
            model[prefix].append(suffix)
    return model



def give_most_frequent(word, model, c_alpha):
    alpha = 0.01
    while word:
        if word in model:
            suffix_counts = Counter(model[word])
            max_count = max(suffix_counts.values())
            most_probable_suffixes = [k for k, v in suffix_counts.items() if v == max_count]
            most_probable_suffix = min(most_probable_suffixes, key=lambda x: corpus.index(x))

            probability = c_alpha * (max_count / sum(suffix_counts.values()))
            return most_probable_suffix, probability
        else:
            word = word[1:]
            c_alpha *= alpha
    return None, c_alpha



def give_random(word, model, c_alpha):
    alpha = 0.01
    while word:
        if word in model:
            suffix_weights = Counter(model[word])
            suffixes, weights = zip(*suffix_weights.items())
            weights = np.array(weights)
            if weights.sum() > 0:
                weights_prob = weights/weights.sum()
                most_probable_suffix_index = np.random.choice(len(suffixes), p=weights_prob)
                most_probable_suffix = suffixes[most_probable_suffix_index]


                probability = c_alpha * weights_prob[most_probable_suffix_index]
                return most_probable_suffix, probability
        word = word[1:]
        c_alpha *= alpha
    return None, c_alpha



def finish_sentence(sentence, n, corpus, randomize=False):
    corpus_counter = Counter(corpus)
    most_common_word = corpus_counter.most_common(1)[0][0]
    model = model_generation(corpus, n)
    c_alpha = 1.0
    alpha = 0.01
    while len(sentence) < 10 and not any(sentence[-1].endswith(p) for p in ".?!"):
        prefix = tuple(sentence[-(n-1):])
        current_n = n
        while current_n > 1:
            if randomize:
                next_word, next_score = give_random(prefix, model, c_alpha)
            else:
                next_word, next_score = give_most_frequent(prefix, model, c_alpha)
            if next_word:
                break
            else:
                current_n -= 1
                prefix = prefix[1:]
                c_alpha *= alpha
            if current_n <= 1:
                next_word = most_common_word
                next_score = c_alpha
                break
        else:
            next_word = most_common_word
            next_score = c_alpha
        sentence.append(next_word)
        c_alpha *= next_score
    return sentence


# word = ["she", "was", "not"]
# n = 3

# print(finish_sentence(word, 3, corpus, randomize=False))
# print(finish_sentence(word, 3, corpus, randomize=True))
