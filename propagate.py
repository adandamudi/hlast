from ast import AST, stmt, parse, unparse
from copy import deepcopy
from pathlib import Path

from gumtree import GumTree, python


def main(argv: list[str]):
    lineno, source, dest = argv[1:]
    propagate(int(lineno), Path(source), Path(dest))


def propagate(lineno: int, source: Path, dest: Path):
    tree, target = map(lambda p: parse(p.read_text(), p), [source, dest])
    node = find(tree, lineno=lineno)
    replicate(tree, node, target)
    return unparse(target)


def replicate(tree: AST, node: stmt, target: AST):
    adapter = python.Adapter(tree, target)
    mapping = GumTree(adapter).mapping(tree, target)
    assert tree == adapter.root(node) and isinstance(node, stmt)

    # print('# tree', adapter.dump(tree), '# node', adapter.dump(node),
    #       '# target', adapter.dump(target), '# mappings', sep='\n')
    # for l, r in mapping:
    #     print(f'{adapter.label(l)}: {adapter.value(l)} ' +
    #           f'-> {adapter.label(r)}: {adapter.value(r)}')
    # print()

    if (node, None) in mapping:
        exit('Already in target!')

    parent = adapter.parent(node)
    if parent is None:
        print('> Insert after', unparse(target))
        target.body.append(deepcopy(node))
        return

    preceding = []
    for sibling in adapter.children(parent):
        if id(sibling) == id(node): break
        if isinstance(sibling, stmt):
            preceding.append(sibling)

    for context in reversed(preceding):
        if (context, None) in mapping:
            ref = mapping[context]
            print('> Insert after', unparse(ref))
            return

    if (parent, None) in mapping:
        ref = mapping[parent]
        print('> Insert before', unparse(ref.body[0]))
        ref.body.insert(0, deepcopy(node))
        return

    exit('Unable to replicate!')


def find(t: AST, *, lineno: int):
    adapter = python.Adapter(t, t)  # FIXME: odd dependency
    res = None
    for n in adapter.postorder(t):
        if getattr(n, 'lineno', lineno) == lineno and isinstance(n, stmt):
            res = n
    return res


if __name__ == "__main__":
    import sys
    main(sys.argv)
