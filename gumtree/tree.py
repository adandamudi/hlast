from dataclasses import dataclass, field
from config import Config


@dataclass(frozen=True)
class Node:
    label: str
    value: str = ''
    children: list['Node'] = field(default_factory=list)
    parent: 'Node' = field(init=False, compare=False, repr=False)

    def __post_init__(self):
        for child in self.children:
            object.__setattr__(child, 'parent', self)
        super().__setattr__('parent', None)
    
    def __getitem__(self, i):  # testing
        return self.children[i]

Tree = Node


class TreeConfig(Config[Tree]):
    pass

config = TreeConfig()
