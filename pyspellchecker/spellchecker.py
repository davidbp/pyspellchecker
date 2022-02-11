import re
import itertools
from collections import Counter

from nltk.collocations import BigramCollocationFinder
from editdistance import eval as edit_distance
from .bktree import BKTree


class SpellChecker:
    def __init__(
            self,
            ngram_range=(1, 2),
            tokenizer=None,
            vocabulary={},
            string_preprocessor_func=str.lower,
            token_pattern=r"(?u)\b\w+|\?",
            lambda_interpolation=0.3,
            min_freq=5,
            max_dist=1,
            sort_candidates=False,
            bktree=True,
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
        """Return a function that splits a string into a sequence of tokens.
        tokenizer: callable
        A function to split a string into a sequence of tokens.
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

    def fit(self, X):
        """
        X iterable of strings (corpus)
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

    def get_candidates(self, token, max_dist):
        if self.bktree:
            return self.get_candidates_bktree(token, max_dist)
        else:
            return self.get_candidates_exhaustive(token, max_dist)

    def get_candidates_bktree(self, token, max_dist):
        """Return a list of candidate words from the vocabulary at most `max_dist` away from the input token."""
        candidate_tokens = self.bktree.query(token, max_dist)

        if len(candidate_tokens) > 0:
            return candidate_tokens
        else:
            return [token]

    def get_candidates_exhaustive(self, token, max_dist):
        """
        Return a list of candidate words from the vocabulary at most `max_dist` away from the input token.
        This version of the function is private and kept for benchmarking pourposes. This function computes the
        edit dist_func between the input token and all words in the vocabulary. Then it filters candidates by
        the edit dist_func.
        """
        token = token.lower()
        distance_token_to_words = {
            word: edit_distance(word, token) for word in self.vocabulary
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
            return [token]

    def filter_vocabulary(self, min_freq):
        self.vocabulary = set(
            dict(
                filter(lambda x: x[1] > min_freq, self.unigram_freq_dict.items())
            ).keys()
        )

    def correct_with_bigrams(self, tokenized_sentence):

        def prob_word(word):
            return self.unigram_freq_dict.get(word, 0) / len(self.unigram_freq_dict)

        def bigrams_starting_by(word):
            return [t for t in list(self.bigram_freq_dict.keys()) if t[0] == word]

        def count_bigrams(list_bigrams):
            return sum(
                [self.bigram_freq_dict.get(bigram, 0) for bigram in list_bigrams]
            )

        def probability_bigram(word1, word2, bigram_freq_dict):
            bigram = (word1, word2)
            if self.bigram_freq_dict.get(bigram, 0) == 0:
                return 0
            else:
                return self.bigram_freq_dict.get(bigram, 0) / count_bigrams(
                    bigrams_starting_by(word1)
                )

        def interpolation_probability(word1, word2):
            return (1 - self.lambda_interpolation) * probability_bigram(
                word1, word2, self.bigram_freq_dict
            ) + self.lambda_interpolation * prob_word(word2)

        for index, word in enumerate(tokenized_sentence):
            if word not in self.vocabulary:
                if index == 0:
                    previous_word = '.'
                else:
                    previous_word = tokenized_sentence[index - 1]

                if word in {'?', '!','b'}:
                    continue
                candidates = {
                    candidate: interpolation_probability(previous_word, candidate)
                    for candidate in self.get_candidates(word, max_dist=self.max_dist)
                }

                tokenized_sentence[index] = max(candidates, key=candidates.get)

        return tokenized_sentence

    def transform(self, x, tokenize=True):
        """Corrects the misspelled words on each element from X.
        X iterable of strings
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
