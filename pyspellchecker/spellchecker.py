import re
import itertools
from collections import Counter
from typing import List, Callable, Optional, Tuple, Iterable

from nltk.collocations import BigramCollocationFinder
from editdistance import eval as edit_distance
from bktree import BKTree


class SpellChecker:
    """
    Basic class to instantiate a spellchecker.

    :param ngram_range: range of the n-gram model.
    :param tokenizer: tokenizer function to convert a string to a list of tokens.
    :param vocabulary: vocabulary used to find corrected words.
    :param string_preprocessor_func: method to preprocess input strings. Defaults as `str.lower`.
    :param token_pattern: str defining a regular expression used to tokenize.
    :param lambda_interpolation: interpolation coefficient for not seen n-grams.
    :param min_freq: minimum frequency needed to store a word in the vocabulary.
    :param max_dist: maximum distance allowed between a misspelled word and a candidate word.
    :param sort_candidates: boolean flag, if true candidates are sorted.
    :param bktree: boolean flag, if true a bktree is used to search candidates.
    """
    def __init__(
            self,
            ngram_range: Tuple[int] = (1, 2),
            tokenizer: Optional[Callable] = None,
            vocabulary: set = {},
            string_preprocessor_func: Callable = str.lower,
            token_pattern: str = r"(?u)\b\w+|\?",
            lambda_interpolation: float = 0.3,
            min_freq: int = 5,
            max_dist: int = 1,
            sort_candidates: bool = False,
            bktree: bool = True,
    ):

        self.ngram_range = ngram_range
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.string_preprocessor_func = string_preprocessor_func
        self.token_pattern = token_pattern
        self.lambda_interpolation = lambda_interpolation
        self.min_freq = min_freq
        self.max_dist = max_dist
        self.sort_candidates = sort_candidates
        self.bktree = bktree

    def build_tokenizer(self):
        """
        Build a function that splits a string into a sequence of tokens.

        If `self.tokenizer` is not None, return it. Otherwise, build a tokenizer
        from a regular expression stored in `self.token_pattern`.
        """
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)

        if token_pattern.groups > 1:
            raise ValueError(
                "More than 1 capturing group in token pattern. Only a single "
                "group should be captured."
            )

        return token_pattern.findall

    def fit(self, X: Iterable[str]):
        """
        Updates the parameters of the spell checker with the data in X.

        :param X: Corpus, represented as an iterable of strings.
        """
        self.tokenize_func = self.build_tokenizer()
        X_tokenized = [self.tokenize_func(self.string_preprocessor_func(x)) for x in X]
        self.unigram_freq_dict = dict(Counter(itertools.chain(*X_tokenized)))
        bigram_finder = BigramCollocationFinder.from_documents(X_tokenized)
        self.bigram_freq_dict = dict(bigram_finder.ngram_fd.items())
        self.vocabulary = set(list(itertools.chain(*self.bigram_freq_dict.keys())))

        if self.min_freq > 0:
            self.filter_vocabulary(min_freq=self.min_freq)

        if self.bktree:
            self.bktree = BKTree(
                edit_distance, self.vocabulary, sort_candidates=self.sort_candidates
            )

    def get_candidates(self, word: str, max_dist: int):
        """
        Finds the candidates for `word` that are at most at distance `max_dist`.
        If `self.bktree` is present it is used to find candidates.

        :param word: word that needs to be corrected.
        :param max_dist: maximum distance from `word` to a candidate.
        :return: list of candidate words.
        """
        if self.bktree:
            return self.get_candidates_bktree(word, max_dist)
        else:
            return self.get_candidates_exhaustive(word, max_dist)

    def get_candidates_bktree(self, word: str, max_dist: int):
        """
        Return a list of candidate words from the vocabulary at most `max_dist` away from the input token
        leveraging a bktree.

        :param word: word that needs to be corrected.
        :param max_dist: maximum distance from `word` to a candidate.
        :return:
        """
        candidate_words = self.bktree.query(word, max_dist)

        if len(candidate_words) > 0:
            return candidate_words
        else:
            return [word]

    def get_candidates_exhaustive(self, word:str , max_dist: int):
        """
        Compute the candidate words from the vocabulary at most `max_dist` away from the input token.
        This function computes the `dist_func` between the input token and all words in the vocabulary.
        Then it filters candidates by the computed values in `dist_func`.
        This function is mainly used for benchmarking purposes.

        :param word: input query word.
        :param max_dist: maximum allowed distance.
        :return:list of candidate words.
        """

        word = word.lower()
        distance_token_to_words = {
            word: edit_distance(word, word) for word in self.vocabulary
        }
        min_dist = min(distance_token_to_words.values())
        if min_dist <= max_dist:

            if self.sort_candidates:
                result = sorted(
                    [
                        (distance, word)
                        for word, distance in distance_token_to_words.items()
                        if distance <= max_dist
                    ]
                )
            else:
                result = [
                    word
                    for word, distance in distance_token_to_words.items()
                    if distance <= max_dist
                ]

            return result
        else:
            return [word]

    def filter_vocabulary(self, min_freq: int):
        """
        Filter words from the vocabulary that have lower than `min_freq` counts.

        :param min_freq: minimum frequency to take into consideration.
        :return:
        """

        self.vocabulary = set(
            dict(
                filter(lambda x: x[1] > min_freq, self.unigram_freq_dict.items())
            ).keys()
        )

    def correct_with_bigrams(self, tokenized_sentence: List[str]):
        """
        Corrects a list of words, one at a time.

        :param tokenized_sentence: list of words.
        :return: list of words containing (possibly) corrections.
        """

        def prob_word(word: str):
            return self.unigram_freq_dict.get(word, 0) / len(self.unigram_freq_dict)

        def bigrams_starting_by(word: str):
            return [t for t in list(self.bigram_freq_dict.keys()) if t[0] == word]

        def count_bigrams(list_bigrams):
            return sum(
                [self.bigram_freq_dict.get(bigram, 0) for bigram in list_bigrams]
            )

        def probability_bigram(word1: str, word2: str):
            bigram = (word1, word2)
            if self.bigram_freq_dict.get(bigram, 0) == 0:
                return 0
            else:
                return self.bigram_freq_dict.get(bigram, 0) / count_bigrams(
                    bigrams_starting_by(word1)
                )

        def interpolation_probability(word1: str, word2: str):
            return (1 - self.lambda_interpolation) * probability_bigram(
                word1, word2, self.bigram_freq_dict
            ) + self.lambda_interpolation * prob_word(word2)

        for index, word in enumerate(tokenized_sentence):
            if word not in self.vocabulary:
                if index == 0:
                    previous_word = '.'
                else:
                    previous_word = tokenized_sentence[index - 1]

                if word in {'?', '!'}:
                    continue
                candidates = {
                    candidate: interpolation_probability(previous_word, candidate)
                    for candidate in self.get_candidates(word, max_dist=self.max_dist)
                }

                tokenized_sentence[index] = max(candidates, key=candidates.get)

        return tokenized_sentence

    def transform(self, x: Iterable[str], tokenize: bool = True):
        """
        Correct the misspelled words on each element in x.

        :param x: Iterable of strings
        :param tokenize: tokenization flag. If true it tokenizes x.
        :return: tokenized sentences
        """
        x = self.string_preprocessor_func(x)

        if tokenize:
            tokenized_sentence = self.tokenize_func(x)
        else:
            tokenized_sentence = x.copy()

        tokenized_sentence = self.correct_with_bigrams(tokenized_sentence)

        return tokenized_sentence

    def save(self, path):
        """Save the instance an PyNgramSpell to the provided path."""
        pass
