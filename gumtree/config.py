from apted import Config as AptedConfig
from typing import Protocol, Iterable, TypeVar


def optimize(orig):
    memo = {}
    def new(self, n):
        if id(n) not in memo:
            memo[id(n)] = orig(self, n)
        return memo[id(n)]
    new.__doc__ = orig.__doc__
    new.__annotations__ = orig.__annotations__
    return new


N = TypeVar('N')


class Config(Protocol[N]):
    def parent(n: N) -> N:
        return n.parent

    def children(n: N) -> Iterable[N]:
        return n.children

    def label(n: N) -> str:
        return n.label

    @optimize
    def height(self, n: N) -> int:
        return 1 + max(map(self.height, self.children(n)), default=0)

    def postorder(self, n: N) -> Iterable[N]:
        for child in self.children(n):
            yield from self.postorder(child)
        yield n

    def descendants(self, n: N) -> Iterable[N]:
        for child in self.children(n):
            yield from self.descendants(child)
            yield child
    
    @optimize
    def num_descendants(self, n: N) -> int:
        return sum(1 for _ in self.descendants(n))
    
    def isomorphic(n1: N, n2: N) -> bool:
        return n1 == n2
