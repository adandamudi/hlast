from abc import abstractmethod
from itertools import zip_longest
from typing import Any, Callable, Iterable, Optional, Protocol, TypeVar

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object


Node = TypeVar('N')


def memoize(orig: Callable[[Node], int]):
    memo = {}

    def new(self, n: Node) -> int:
        if id(n) not in memo:
            memo[id(n)] = orig(self, n)
        return memo[id(n)]

    return new


class BaseAdapter(Protocol[Node]):
    # Implement these for your tree implementation!

    @abstractmethod
    def parent(self, n: Node) -> Optional[Node]:
        raise NotImplementedError

    @abstractmethod
    def children(self, n: Node) -> Iterable[Node]:
        raise NotImplementedError

    @abstractmethod
    def label(self, n: Node) -> str:
        raise NotImplementedError

    @abstractmethod
    def value(self, n: Node) -> Any:
        raise NotImplementedError

    # These should just work, but could be optimized!

    @memoize
    def height(self, n: Node) -> int:
        return 1 + max(map(self.height, self.children(n)), default=0)

    @memoize
    def num_descendants(self, n: Node) -> int:
        return sum(1 + self.num_descendants(c) for c in self.children(n))

    def contains(self, n: Node, t: Node) -> bool:
        while parent := self.parent(n):
            if id(parent) == id(t):
                return True
            n = parent
        return False

    def isomorphic(self, n1: Node, n2: Node) -> bool:
        return all(prop(n1) == prop(n2) for prop in
                   [self.label, self.value, self.height, self.num_descendants]) \
            and all(self.isomorphic(c1, c2) for c1, c2 in
                    zip_longest(*map(self.children, [n1, n2])))

    # These are unlikely to benefit from optimization

    def descendants(self, n: Node) -> Iterable[Node]:
        for child in self.children(n):
            yield from self.descendants(child)
            yield child

    def postorder(self, n: Node) -> Iterable[Node]:
        for child in self.children(n):
            yield from self.postorder(child)
        yield n

    # These are just a debugging / assertion aids

    def root(self, n: Node) -> Node:
        while parent := self.parent(n):
            n = parent
        return n

    def dump(self, n: Node, indent=0):
        return '\n'.join(['\t'*indent + f'{self.label(n)}: {self.value(n)}',
                          *(self.dump(c, indent+1) for c in self.children(n))])


# Type export
Adapter = BaseAdapter
