from gumtree import GumTree, Mapping
from tree import Tree, Node, config


N = Node
gt = GumTree(config)


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
                                N('StringLiteral', 'Bar'),
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
    topdown = Mapping(config)
    topdown.add_subtree(source[0][2][1], destination[0][2][1])
    topdown.add_subtree(source[0][2][3], destination[0][2][3])
    topdown.add_subtree(source[0][2][4][0][0], destination[0][2][4][0][0])
    topdown.add_subtree(source[0][2][4][0][1], destination[0][2][4][0][2][1])
    assert match(topdown, gt.topdown(source, destination))

    bottomup = Mapping(config, topdown)
    bottomup.add(source[0][2][4][0], destination[0][2][4][0])
    bottomup.add(source[0][2][4], destination[0][2][4])
    bottomup.add(source[0][2], destination[0][2])
    bottomup.add(source[0][2][0], destination[0][2][0])
    bottomup.add(source[0][2][2], destination[0][2][2])
    bottomup.add(source[0], destination[0])
    bottomup.add(source[0][0], destination[0][0])
    bottomup.add(source[0][1], destination[0][1])
    bottomup.add(source, destination)
    assert match(bottomup, gt.bottomup(source, destination, topdown))

    expected = bottomup
    assert match(expected, gt.mapping(source, destination))
    print('Passed Example!')


def match(left: Mapping, right: Mapping):
    return all(id(el) == id(rl) and id(er) == id(rr)
               for (el, er), (rl, rr) in zip(left, right))


if __name__ == "__main__":
    example()
