from ast import AST, Name, parse, unparse, walk, stmt
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from gumtree import GumTree, Mapping, python

# https://github.com/PyCQA/pylint/issues/3882
# pylint: disable=unsubscriptable-object


def main(argv: list[str]):
    lineno, src, dst, out = argv[1:]
    propagate(int(lineno), Path(src), Path(dst), Path(out))


def propagate(lineno: int, src: Path, dst: Path, out: Path):
    tree, target = map(lambda p: parse(p.read_text(), p), [src, dst])
    replicate(tree, find(tree, lineno=lineno), target)
    out.write_text(unparse(target) + '\n')


def replicate(tree: AST, node: stmt, target: AST):
    adapter = python.Adapter(tree, target)
    mapping = GumTree(adapter).mapping(tree, target)
    assert tree == adapter.root(node) and isinstance(node, stmt)

    # print('# TREE', adapter.dump(tree),
    #       '# TARGET', adapter.dump(target), sep='\n')
    # print('# MAPPINGS', '\n'.join('\t->\t'.join(adapter.label(n)
    #       for n in (l, r)) for l, r in mapping), sep='\n')

    if (node, None) in mapping:
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
        if (context, None) in mapping:
            reference = mapping[context]
            block = adapter.parent(reference)
            new = adapt(node, tree, mapping)
            block.insert(1 + block.index(reference), new)
            return

    if (parent, None) in mapping:
        block = mapping[parent]
        new = adapt(node, tree, mapping)
        block.insert(0, new)
        return

    exit('Unable to replicate!')


def adapt(node: AST, tree: AST, mapping: Mapping):
    # Count all renames detected by GumTree
    count = defaultdict(lambda: defaultdict(int))
    for n in walk(tree):
        if isinstance(n, Name) and (n, None) in mapping:
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
    import sys
    main(sys.argv)
