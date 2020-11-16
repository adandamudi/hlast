#!/usr/bin/env python

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object

from itertools import product
from collections.abc import Iterable
from typing import Generic, Optional, TypeVar
from bidict import bidict

from config import Config
from priorityq import PriorityQ


Tree = TypeVar('Tree')

class HeightPQ(PriorityQ[Tree, int]):
    def __init__(self, config: Config[Tree], it=[]):
        super().__init__(it, key=config.height, reverse=True)
        self.config = config

    def peek_max(self) -> int:
        return len(self) and self.config.height(self.peek())

    def open(self, tree: Tree):
        for child in self.config.children(tree):
            self.push(child)

    def pop(self) -> Tree:
        trees = []
        assert self, 'Empty!'
        height = self.peek_max()
        while self.peek_max() == height:
            trees.append(super().pop())
        return trees


class Mapping(Generic[Tree]):
    def __init__(self, config: Config[Tree], m: Iterable[tuple[Tree, Tree]] = ()):
        self.config = config
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
        for n1, n2 in zip(*map(self.config.postorder, [t1, t2])):
            self.add(n1, n2)

    def mapped(self, t1: Optional[Tree], t2: Optional[Tree]) -> bool:
        if t1 is not None and t2 is not None:
            return self._m[id(t1)] == id(t2)
        if t2 is not None:
            return id(t2) in self._m.inverse
        if t1 is not None:
            return id(t1) in self._m
        assert False, "Both parameters are None!"

    def __iter__(self) -> Iterable[tuple[Tree, Tree]]:
        for l, r in self._m.items():
            yield self._l[l], self._r[r]


class GumTree(Generic[Tree]):
    defaults = {'min_height': 2, 'min_dice': .50, 'max_size': 100}

    def __init__(self, config: Config[Tree], *, opt=None, **params):
        assert not set(params) - set(self.defaults), 'Invalid parameters!'
        self.params = dict(self.defaults, **params)
        self.opt = opt or self.apted
        self.config = config

    def mapping(self, t1: Tree, t2: Tree):
        return self.bottomup(t1, t2, self.topdown(t1, t2))

    def topdown(self, t1: Tree, t2: Tree) -> Mapping[Tree]:
        min_height = self.params['min_height']
        c = self.config

        l1, l2 = HeightPQ(c, [t1]), HeightPQ(c, [t2])
        a, m = Mapping(c), Mapping(c)

        # Note: Algorithm uses >, but the example seems to use >= instead.
        while max(l.peek_max() for l in [l1, l2]) >= min_height:
            if l1.peek_max() != l2.peek_max():
                pq = max(l1, l2, key=HeightPQ.peek_max)
                for t in pq.pop():
                    pq.open(t)
            else:
                h1, h2 = l1.pop(), l2.pop()
                for n1, n2 in product(h1, h2):
                    if c.isomorphic(n1, n2):
                        if h2.count(n1) > 1 or h1.count(n2) > 1:
                            a.add(n1, n2)
                        else:
                            m.add_subtree(n1, n2)
                for t1 in h1:
                    if m.mapped(t1, None):
                        l1.open(t1)
                for t2 in h2:
                    if m.mapped(None, t2):
                        l2.open(t2)
        
        def dice(ts):
            t1, t2 = ts
            return self.dice(c.parent(t1), c.parent(t2), m)

        # Note: Algorithm removes from A, but I think this is more efficient
        for n1, n2 in sorted(a, key=dice, reverse=True):
            if not m.mapped(n1, None) and not m.mapped(None, n2):
                m.add_subtree(n1, n2)

        return m

    def bottomup(self, t1: Tree, t2: Tree, m: Mapping[Tree]) -> Mapping[Tree]:
        min_dice = self.params['min_dice']
        max_size = self.params['max_size']
        c = self.config

        # FIXME: Paper mentions candidates must have descendants matched, but
        # it was hard to do and if min_dice > 0 they will be dropped anyways.
        assert min_dice > 0

        def candidate(n1: Tree, m: Mapping[Tree]):
            return max((c2 for c2 in c.postorder(t2)
                        if c.label(n1) == c.label(c2) and not m.mapped(None, c2)),
                       key=lambda c2: self.dice(n1, c2, m), default=None)

        for n1 in c.postorder(t1):
            if not m.mapped(n1, None):
                n2 = candidate(n1, m)
                if n2 and self.dice(n1, n2, m) > min_dice:
                    m.add(n1, n2)
                    if max(c.num_descendants(t) for t in [n1, n2]) < max_size:
                        # Note: Paper mentions removing already matched descendants
                        for ta, tb in self.opt(n1, n2):
                            if ta is not None and tb is not None \
                                    and not m.mapped(ta, None) and not m.mapped(None, tb) \
                                    and c.label(ta) == c.label(tb):
                                m.add(ta, tb)
        return m

    def dice(self, t1: Tree, t2: Tree, m: Mapping[Tree]):
        s, ns = self.config.descendants, self.config.num_descendants

        # Note: Formula is unclear, I think this is what they meant ¯\_(ツ)_/¯
        return (2 * sum(1 for n1, n2 in m if n1 in s(t1) and n2 in s(t2))
                / (ns(t1) + ns(t2) or 1))  # avoid division by 0

    def apted(self, t1: Tree, t2: Tree):
        from apted import APTED, Config

        gt = self

        class CustomConfig(Config):
            def rename(self, n1: Tree, n2: Tree):
                return int(gt.config.label(n1) != gt.config.label(n2))

            def children(self, n: Tree):
                return list(gt.config.children(n))

        return APTED(t1, t2, CustomConfig()).compute_edit_mapping()
