#!/usr/bin/env python

from argparse import ArgumentParser, FileType, Namespace
from ast import AST, Name, parse, unparse, walk, stmt, literal_eval
from collections import defaultdict
from copy import deepcopy
from typing import TextIO
import sys

from gumtree import GumTree, Mapping, python

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object


def add_arguments(parser: ArgumentParser):
    parser.add_argument('lineno', type=int)
    parser.add_argument('source', type=FileType('r'))
    parser.add_argument('target', type=FileType('r+'))
    parser.add_argument('--out', type=FileType('w'), default=sys.stdout)
    parser.add_argument('--minor', type=int, default=sys.version_info[1])
    parser.add_argument('--gumtree', type=literal_eval, default='{}')
    return parser


def propagate(args: Namespace):
    tree, target = [parse(f.read(), feature_version=(3, args.minor))
                    for f in (args.source, args.target)]
    replicate(tree, find(tree, lineno=args.lineno), target, **args.gumtree)
    print(unparse(target), file=args.out)


def replicate(tree: AST, node: stmt, target: AST, **kwargs):
    adapter = python.Adapter(tree, target)
    mapping = GumTree(adapter, **kwargs).mapping(tree, target)
    assert tree == adapter.root(node) and isinstance(node, stmt)

    # print('# TREE', adapter.dump(tree),
    #       '# TARGET', adapter.dump(target), sep='\n')
    # print('# MAPPING', '\n'.join('\t->\t'.join(adapter.label(n)
    #       for n in (l, r)) for l, r in mapping.items()), sep='\n')

    if node in mapping:
        exit('Already in target!')

    parent = adapter.parent(node)
    if parent is None:
        target.body.append(deepcopy(node))
        return

    preceding = []
    for sibling in adapter.children(parent):
        if id(sibling) == id(node):
            break
        if isinstance(sibling, stmt):
            preceding.append(sibling)

    for context in reversed(preceding):
        if context in mapping:
            reference = mapping[context]
            block = adapter.parent(reference)
            new = adapt(node, tree, mapping)
            block.insert(1 + block.index(reference), new)
            return

    if parent in mapping:
        block = mapping[parent]
        new = adapt(node, tree, mapping)
        block.insert(0, new)
        return

    exit('Unable to replicate!')


def adapt(node: AST, tree: AST, mapping: Mapping):
    # Count all renames detected by GumTree
    count = defaultdict(lambda: defaultdict(int))
    for n in walk(tree):
        if isinstance(n, Name) and n in mapping:
            count[n.id][mapping[n].id] += 1
    # Select the most common as canonical
    renames = {orig: max(options, key=options.get)
               for orig, options in count.items()}
    # Update the names in given node
    new = deepcopy(node)
    for n in walk(new):
        if isinstance(n, Name) and n.id in renames:
            n.id = renames[n.id]
    return new


def find(t: AST, *, lineno: int):
    adapter = python.Adapter(t)  # FIXME: odd dependency
    res = None
    for n in adapter.postorder(t):
        if getattr(n, 'lineno', lineno) == lineno and isinstance(n, stmt):
            res = n
    return res


if __name__ == "__main__":
    propagate(add_arguments(ArgumentParser()).parse_args(sys.argv[1:]))
