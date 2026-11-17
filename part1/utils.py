import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # AI usage: ChatGPT generated neighbors (typo case) and excluded words (synonym replacement case)
    keyboard_neighbors = {
        "a": "qwsz", "b": "vghn", "c": "xdfv",
        "d": "serfcx", "e": "wsdr", "f": "drtgvc",
        "g": "ftyhbv", "h": "gyujnb", "i": "ujko",
        "j": "huikmn", "k": "jiolm", "l": "kop",
        "m": "njk", "n": "bhjm", "o": "iklp",
        "p": "ol", "q": "wa", "r": "edft",
        "s": "awedxz", "t": "rfgy", "u": "yhji",
        "v": "cfgb", "w": "qase", "x": "zsdc",
        "y": "tghu", "z": "asx"
    }

    excluded_words = {
    "not", "no", "never", "nor",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "the", "and", "or", "but",
    "in", "on", "at", "to", "of", "for", "with", "by",
    "i", "you", "he", "she", "it", "we", "they",
    "this", "that", "these", "those",
    "have", "has", "had", "do", "does", "did"
}

    # split into words
    words = example["text"].split()
    transformed_words = []

    # typo and synonym replacement prob
    typo_prob = 0.2
    synonym_prob = 0.1
    for word in words:
        r = random.random()
        # typo case
        # find positions which contain characters that have entries in keyboard_neighbors
        if r < typo_prob:
            positions = [
                i for i, char in enumerate(word)
                if char.lower() in keyboard_neighbors
            ]
            if positions:
                # randomly select a charactor position and randomly choose a neighbor of that character
                i = random.choice(positions)
                original_char = word[i]
                replacement = random.choice(
                    keyboard_neighbors[original_char.lower()]
                )
                if original_char.isupper():
                    replacement = replacement.upper()

                # replace the one character in word
                word = word[:i] + replacement + word[i+1:]

        # synonym case
        elif r < typo_prob + synonym_prob: # separate 10% prob synonym replacement
            # ChatGPT suggested to "Restrict to alphabetic words and avoid function words"
            if word.isalpha() and word.lower() not in excluded_words:
                # Synset: a set of synonyms that share a common meaning
                synsets = wordnet.synsets(word.lower())
                if synsets:
                    synonyms = [
                        lemma.name()
                        for lemma in synsets[0].lemmas() # pick the first synset and get its lemmas
                        if lemma.name().isalpha() and lemma.name().lower() != word.lower()
                    ]
                    if synonyms:
                        replacement = random.choice(synonyms)
                        if word.isupper():
                            replacement = replacement.upper()
                        elif word[0].isupper():
                            replacement = replacement.capitalize()

                        word = replacement

        transformed_words.append(word)

    # Old: typo only version
    # for word in words:
    #     if random.random() < 0.2: # 0.2 prob of being modified (typo)
    #         # find positions which contain characters that have entries in keyboard_neighbors
    #         positions = [
    #             i for i, char in enumerate(word)
    #             if char.lower() in keyboard_neighbors
    #         ]
    #         if positions:
    #             # randomly select a charactor position and randomly choose a neighbor of that character
    #             i = random.choice(positions)
    #             original_char = word[i]
    #             replacement = random.choice(
    #                 keyboard_neighbors[original_char.lower()]
    #             )
    #             if original_char.isupper():
    #                 replacement = replacement.upper()

    #             # replace the one character in word
    #             word = word[:i] + replacement + word[i+1:]

    #     # append original or modified word
    #     transformed_words.append(word)

    example["text"] = " ".join(transformed_words)

    # raise NotImplementedError

    ##### YOUR CODE ENDS HERE ######

    return example
