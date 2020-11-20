from abc import abstractmethod
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

    # These should just "work", but could be optimized!

    @memoize
    def height(self, n: Node) -> int:
        return 1 + max(map(self.height, self.children(n)), default=0)

    @memoize
    def num_descendants(self, n: Node) -> int:
        return sum(1 for _ in self.postorder(n)) - 1

    def postorder(self, n: Node) -> Iterable[Node]:
        for child in self.children(n):
            yield from self.postorder(child)
        yield n

    def isomorphic(self, n1: Node, n2: Node) -> bool:
        return (self.label(n1) == self.label(n2)
                and self.value(n1) == self.value(n2)
                and self.height(n1) == self.height(n2)
                and all(self.isomorphic(c1, c2) for c1, c2
                        in zip(*map(self.children, [n1, n2]))))

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
