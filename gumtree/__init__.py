from apted import APTED, Config
from itertools import product, zip_longest
from collections.abc import Iterable
from typing import Generic, Optional, TypeVar
from bidict import bidict

from .adapter import Adapter
from .priorityq import PriorityQ

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object


Tree = TypeVar('Tree')


class HeightPQ(PriorityQ[Tree, int]):
    def __init__(self, adapter: Adapter[Tree], it=[]):
        super().__init__(it, key=adapter.height, reverse=True)
        self.adapter = adapter

    def peek_max(self) -> int:
        return len(self) and self.adapter.height(self.peek())

    def open(self, tree: Tree):
        for child in self.adapter.children(tree):
            self.push(child)

    def pop(self) -> Tree:
        trees = []
        assert self, 'Empty!'
        height = self.peek_max()
        while self.peek_max() == height:
            trees.append(super().pop())
        return trees


class Mapping(Generic[Tree]):
    def __init__(self, adapter: Adapter[Tree], m: Iterable[tuple[Tree, Tree]] = ()):
        self.adapter = adapter
        self._m = bidict()
        self._l = {}
        self._r = {}
        for ts in m:
            self.add(*ts)

    def add(self, n1: Tree, n2: Tree):
        self._m.put(id(n1), id(n2))
        self._l[id(n1)] = n1
        self._r[id(n2)] = n2

    def add_subtree(self, t1: Tree, t2: Tree):
        for n1, n2 in zip_longest(*map(self.adapter.postorder, [t1, t2])):
            if (n1, None) not in self and (None, n2) not in self:
                self.add(n1, n2)
            assert (n1, n2) in self

    def __contains__(self, ts: tuple[Optional[Tree], Optional[Tree]]) -> bool:
        if all(t is not None for t in ts):
            return self._m.get(id(ts[0]), None) == id(ts[1])
        if ts[1] is not None:
            return id(ts[1]) in self._m.inverse
        if ts[0] is not None:
            return id(ts[0]) in self._m
        assert False, "Both parameters are None!"

    def __iter__(self) -> Iterable[tuple[Tree, Tree]]:
        for l, r in self._m.items():
            yield self._l[l], self._r[r]

    def __getitem__(self, t1: Tree) -> Tree:
        return self._r[self._m[id(t1)]]

    @property
    def inv(self):
        m = Mapping(self.adapter)
        m._l, m._r = m._r, m._l
        m._m = self._m.inv
        return m


class GumTree(Generic[Tree]):
    defaults = {'min_height': 2, 'min_dice': .50, 'max_size': 100}

    def __init__(self, adapter: Adapter[Tree], *, opt=None, **params):
        assert not set(params) - set(self.defaults), 'Invalid parameters!'
        self.params = dict(self.defaults, **params)
        self.opt = opt or self.apted
        self.adapter = adapter

    def mapping(self, t1: Tree, t2: Tree):
        return self.bottomup(t1, t2, self.topdown(t1, t2))

    def topdown(self, t1: Tree, t2: Tree) -> Mapping[Tree]:
        min_height = self.params['min_height']

        adapt = self.adapter
        parent = self.adapter.parent
        isomorphic = self.adapter.isomorphic
        def different(l, r): return id(l) != id(r)

        l1, l2 = HeightPQ(adapt, [t1]), HeightPQ(adapt, [t2])
        a, m = [], Mapping(adapt)

        # Note: Algorithm uses >, but the example seems to use >= instead.
        while max(l.peek_max() for l in [l1, l2]) >= min_height:
            if l1.peek_max() != l2.peek_max():
                pq = max(l1, l2, key=HeightPQ.peek_max)
                for t in pq.pop():
                    pq.open(t)
            else:
                h1, h2 = l1.pop(), l2.pop()
                for n1, n2 in product(h1, h2):
                    if isomorphic(n1, n2):
                        if any(isomorphic(n1, t) for t in h2 if different(t, n2)) or \
                           any(isomorphic(t, n2) for t in h1 if different(t, n1)):
                            a.append((n1, n2))
                        else:
                            m.add_subtree(n1, n2)
                for t1 in h1:
                    if (t1, None) not in m:
                        l1.open(t1)
                for t2 in h2:
                    if (None, t2) not in m:
                        l2.open(t2)

        a.sort(key=lambda ts: self.dice(*map(parent, ts), m), reverse=True)
        for n1, n2 in a:
            # Note: Algorithm removes from A, but I think this is more efficient
            if (n1, None) not in m and (None, n2) not in m:
                m.add_subtree(n1, n2)

        return m

    def bottomup(self, t1: Tree, t2: Tree, m: Mapping[Tree]) -> Mapping[Tree]:
        min_dice = self.params['min_dice']
        max_size = self.params['max_size']

        label = self.adapter.label
        postorder = self.adapter.postorder
        num_descendants = self.adapter.num_descendants

        # FIXME: Paper mentions candidates must have descendants matched, but
        # it was hard to do and if min_dice > 0 they will be dropped anyways.
        assert min_dice > 0

        def candidate(n1: Tree, m: Mapping[Tree]):
            return max((c2 for c2 in postorder(t2)
                        if label(n1) == label(c2) and (None, c2) not in m),
                       key=lambda c2: self.dice(n1, c2, m), default=None)

        for n1 in postorder(t1):
            if (n1, None) not in m:
                n2 = candidate(n1, m)
                if n2 and self.dice(n1, n2, m) > min_dice:
                    m.add(n1, n2)
                    if max(num_descendants(t) for t in [n1, n2]) < max_size:
                        # Note: Paper mentions removing already matched descendants
                        for ta, tb in self.opt(n1, n2):
                            if ta is not None and tb is not None \
                                    and (ta, None) not in m and (None, tb) not in m \
                                    and label(ta) == label(tb):
                                m.add(ta, tb)
        return m

    def dice(self, t1: Tree, t2: Tree, m: Mapping[Tree]):
        num_descendants = self.adapter.num_descendants
        def s(n): return map(id, self.adapter.postorder(n))

        # Note: Formula is unclear, I think this is what they meant ¯\_(ツ)_/¯
        return (2 * sum(1 for n1, n2 in m if id(n1) in s(t1) and id(n2) in s(t2))
                / (num_descendants(t1) + num_descendants(t2) or 1))

    def apted(self, t1: Tree, t2: Tree):
        return APTED(t1, t2, AptedConfig(self.adapter)).compute_edit_mapping()


class AptedConfig(Config):
    def __init__(self, adapter):
        self.adapter = adapter

    def rename(self, n1: Tree, n2: Tree):
        return int(self.adapter.label(n1) != self.adapter.label(n2))

    def children(self, n: Tree):
        return list(self.adapter.children(n))
