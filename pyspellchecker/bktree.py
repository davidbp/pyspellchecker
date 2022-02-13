
from typing import Iterable, Callable, Tuple

class BKTree:
    """
    BKTree for storing and querying objects using discrete distances.

    :param dist_func: dist_func function used to compute dist_func between words.
    :param words: Sequence of words to be stored in the BKTree.
    :param sort_candidates: Boolean flag to sort candidates when `.query` is called.
    """

    def __init__(
        self, dist_func: Callable, words: Iterable[str], sort_candidates: bool = False
    ):
        self.dist_func = dist_func
        self.sort_candidates = sort_candidates

        it = iter(words)
        root = next(it)
        self.tree = (root, {})

        for i in it:
            self._add_word(self.tree, i)

    def _add_word(self, parent: Tuple[str, dict], word: str):
        """Add word to the BKTree.

        :param parent: parent node composed of (word, subtree).
        :param word: word to be added.
        """
        parent_word, children = parent
        d = self.dist_func(word, parent_word)
        if d in children:
            self._add_word(children[d], word)
        else:
            children[d] = (word, {})

    def _search_descendants(
        self,
        parent: Tuple[str, dict],
        max_dist: int,
        dist_func: Callable,
        query_word: str,
    ):
        """
        Retrieve descendants of a parent node.

        :param parent: parent node of the form `(str, dict)`.
        :param max_dist: maximum distance to the parent node.
        :param dist_func: distance function between words.
        :param query_word:
        :return: Descendants of a particular `parent` node.
        """

        node_word, children_dict = parent
        dist_to_node = dist_func(query_word, node_word)
        self.visited_nodes.append(node_word)
        results = []
        if dist_to_node <= max_dist:
            results.append((dist_to_node, node_word))

        for i in range(dist_to_node - max_dist, dist_to_node + max_dist + 1):
            child = children_dict.get(i)
            if child is not None:
                results.extend(
                    self._search_descendants(child, max_dist, dist_func, query_word)
                )

        return results

    def query(
        self, query_word: str, max_dist: int, return_distances: bool = False
    ):
        """
        Search closest to que a query.

        Search all words that are at maximum distance of `max_dist` to the input `query_word` according to `self.dist_func`.
        Words at distance `max_dist` are included in the set of retrieved words.
        In other words, this method returns `[w in bktree if self.dist_func(w, query_word) <= max_dist]`.

        If `return_distances=False` it returns a list of words at distance `max_dist` to the `query_word`.
        If `return_distances=True` it returns a list of (word, distance) pairs with all pairs at `max_dist` the `query_word`.

        :param query_word: word used as query.
        :param max_dist: maximum distance allowed to the query word.
        :param return_distances: boolean flag, if true it returns words and distances.
        :return: all words at distance `max_dist` to the `query_word`.
        """
        self.visited_nodes = []

        distance_candidate_list = self._search_descendants(
            self.tree, max_dist, self.dist_func, query_word
        )
        if self.sort_candidates:
            distance_candidate_list = sorted(distance_candidate_list)

        if return_distances:
            return distance_candidate_list
        else:
            return [x[1] for x in distance_candidate_list]
