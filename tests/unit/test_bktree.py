
import pytest
from pyspellchecker import BKTree
from editdistance import eval as edit_distance

@pytest.mark.parametrize('vocabulary', ['cat','dog','cats','docs'])
def test_bktree_fit(data):
    bktree = BKTree(edit_distance, vocabulary, sort_candidates=False)
    bktree._search_descendants