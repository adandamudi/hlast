from ast import AST, iter_child_nodes
from typing import Any, Iterable, Optional

from .adapter import BaseAdapter, memoize

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object


class Adapter(BaseAdapter[AST]):
    def __init__(self, root1: AST, root2: AST):
        self._parents = {}
        for root in [root1, root2]:
            self._update_parents(root)

    def parent(self, n: AST) -> Optional[AST]:
        return self._parents.get(id(n), None)

    @staticmethod
    def children(n: AST) -> Iterable[AST]:
        return iter_child_nodes(n)

    @staticmethod
    def label(n: AST) -> str:
        return type(n).__name__

    @memoize
    def value(self, n: AST) -> Any:
        terminals = []
        for attr in dir(n):
            if attr.startswith('_') \
                    or 'lineno' in attr or 'col' in attr \
                    or isinstance(v := getattr(n, attr, None), (AST, list)):
                continue  # Skip metadata and AST nodes
            terminals.append((attr, v))
        return terminals

    def _update_parents(self, root):
        for child in self.children(root):
            self._parents[id(child)] = root
            self._update_parents(child)
