import pytest
from pyspellchecker import BKTree
from editdistance import eval as edit_distance


@pytest.mark.parametrize(
    'vocabulary',
    [
        ['single_word'],
        ['cat', 'dog', 'cats', 'docs'],
        ['repeated_word', 'repeated_word', 'cats', 'docs'],
    ],
)
def test_bktree_fit(vocabulary):
    bktree = BKTree(edit_distance, vocabulary)
    assert sorted(bktree.vocabulary) == sorted(vocabulary)
