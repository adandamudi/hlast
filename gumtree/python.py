from ast import AST, iter_fields
from typing import Any, Iterable, Optional, Union, get_args

from .adapter import BaseAdapter, memoize

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object

from ast import expr_context, boolop, operator, unaryop, cmpop
Enums = (expr_context, boolop, operator, unaryop, cmpop)


Node = Union[AST, list]


class Adapter(BaseAdapter[Node]):
    def __init__(self, root1: Node, root2: Node):
        self._parents = {}
        for root in [root1, root2]:
            self._update_parents(root)

    def parent(self, n: Node) -> Optional[Node]:
        return self._parents.get(id(n), None)

    @staticmethod
    def children(n: Node) -> Iterable[Node]:
        if isinstance(n, AST):
            it = iter_fields(n)
        if isinstance(n, list):
            it = enumerate(n)

        for _, value in it:
            if isinstance(value, (AST, list)) and not isinstance(value, Enums):
                yield value

    @staticmethod
    def label(n: Node) -> str:
        return type(n).__name__

    @memoize
    def value(self, n: Node) -> Any:
        terminals = []
        if isinstance(n, AST):
            for name, value in iter_fields(n):
                if isinstance(value, list) and value and all(isinstance(e, Enums) for e in value):
                    terminals.append((name, value))
                if value is not None and not isinstance(value, (AST, list)) or isinstance(value, Enums):
                    terminals.append((name, value))
        return terminals

    def _update_parents(self, root):
        for child in self.children(root):
            assert id(child) not in self._parents
            self._parents[id(child)] = root
            self._update_parents(child)
