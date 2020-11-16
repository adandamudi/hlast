from ast import AST, NodeVisitor, iter_child_nodes
from config import Config, optimize
from typing import Iterable


class ParentTracker(NodeVisitor):
    def __init__(self):
        self.parent = None
        self.parents = {}

    def generic_visit(self, node):
        self.parents[id(self)] = self.parent
        old, self.parent = self.parent, self
        super().generic_visit(node)
        self.parent = old


class AstConfig(Config[AST]):
    def __init__(self, root1, root2):
        tracker = ParentTracker()
        for root in [root1, root2]:
            tracker.visit(root)
        self.parents = tracker.parents

    def parent(self, n: AST) -> AST:
        return self.parents[id(n)]

    def children(self, n: AST) -> Iterable[AST]:
        return iter_child_nodes(n)

    def label(self, n: AST) -> str:
        return type(n).__name__

    def isomorphic(self, n1: AST, n2: AST) -> bool:
        return (self.label(n1) == self.label(n2)
                and self.value(n1) == self.value(n2)
                and all(self.isomorphic(c1, c2) for c1, c2
                in zip(*map(self.descendants, [n1, n2]))))
    
    @optimize
    def value(self, n: AST):
        terminals = []
        for attr in dir(n):
            if not attr.startswith('_'):
                if 'lineno' not in attr and 'col' not in attr:
                    v = getattr(n, attr, None)
                    if not isinstance(v, (AST, list)):
                        print(self.label(n), attr, v)
                        terminals.append(v)
        return terminals