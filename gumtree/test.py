#!/usr/bin/env python

from . import GumTree, Mapping
from .tree import Node, adapter


N = Node
gt = GumTree(adapter)


def test():
    t1, t2 = example()
    topdown = Mapping(adapter)
    topdown.add_subtree(t1[0][2][1], t2[0][2][1])
    topdown.add_subtree(t1[0][2][3], t2[0][2][3])
    topdown.add_subtree(t1[0][2][4][0][0], t2[0][2][4][0][0])
    topdown.add_subtree(t1[0][2][4][0][1], t2[0][2][4][0][2][1])
    assert match(topdown, gt.topdown(t1, t2))

    bottomup = Mapping(adapter, topdown)
    bottomup.add(t1[0][2][4][0], t2[0][2][4][0])
    bottomup.add(t1[0][2][4], t2[0][2][4])
    bottomup.add(t1[0][2], t2[0][2])
    bottomup.add(t1[0][2][0], t2[0][2][0])
    bottomup.add(t1[0][2][2], t2[0][2][2])
    bottomup.add(t1[0], t2[0])
    bottomup.add(t1[0][0], t2[0][0])
    bottomup.add(t1[0][1], t2[0][1])
    bottomup.add(t1, t2)
    assert match(bottomup, gt.bottomup(t1, t2, topdown))

    expected = bottomup
    assert match(expected, gt.mapping(t1, t2))
    print('Passed Example!')


def example():
    source = \
        N('CompilationUnit', '', [
            N('TypeDeclaration', '', [
                N('Modifier', 'public'),
                N('SimpleName', 'Test'),
                N('MethodDeclaration', '', [
                    N('Modifier', 'private'),
                    N('SimpleType', 'String', [
                        N('SimpleName', 'String'),
                    ]),
                    N('SimpleName', 'foo'),
                    N('SingleVariableDeclaration', '', [
                        N('PrimitiveType', 'int'),
                        N('SimpleName', 'i'),
                    ]),
                    N('Block', '', [
                        N('IfStatement', '', [
                            N('InfixExpression', '==', [
                                N('SimpleName', 'i'),
                                N('NumberLiteral', '0'),
                            ]),
                            N('ReturnStatement', '', [
                                N('StringLiteral', 'Foo!'),
                            ]),
                        ]),
                    ]),
                ])
            ])
        ])
    destination = \
        N('CompilationUnit', '', [
            N('TypeDeclaration', '', [
                N('Modifier', 'public'),
                N('SimpleName', 'Test'),
                N('MethodDeclaration', '', [
                    N('Modifier', 'private'),
                    N('SimpleType', 'String', [
                        N('SimpleName', 'String'),
                    ]),
                    N('SimpleName', 'foo'),
                    N('SingleVariableDeclaration', '', [
                        N('PrimitiveType', 'int'),
                        N('SimpleName', 'i'),
                    ]),
                    N('Block', '', [
                        N('IfStatement', '', [
                            N('InfixExpression', '==', [
                                N('SimpleName', 'i'),
                                N('NumberLiteral', '0'),
                            ]),
                            N('ReturnStatement', '', [
                                N('StringLiteral', 'Bar'),
                            ]),
                            N('IfStatement', '', [
                                N('InfixExpression', '==', [
                                    N('SimpleName', 'i'),
                                    N('PrefixExpression', '-', [
                                        N('NumberLiteral', '1'),
                                    ]),
                                ]),
                                N('ReturnStatement', '', [
                                    N('StringLiteral', 'Foo!'),
                                ]),
                            ]),
                        ]),
                    ]),
                ])
            ])
        ])
    return source, destination


def match(left: Mapping, right: Mapping):
    return all(id(el) == id(rl) and id(er) == id(rr)
               for (el, er), (rl, rr) in zip(left, right))


if __name__ == "__main__":
    test()
