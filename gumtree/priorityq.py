from heapq import heapify, heappush, heappop
from typing import Callable, Generic, Iterable, TypeVar
from operator import lt, gt


T, K = TypeVar('T'), TypeVar('K')

class PriorityQ(Generic[T, K]):

    def __init__(self, it: Iterable[T], *, key: Callable[[T], K] = lambda x: x, reverse=False):
        self._Item = self._itemizer(key, reverse)
        self._heap = list(map(self._Item, it))
        heapify(self._heap)

    def push(self, value: T):
        item = self._Item(value)
        heappush(self._heap, item)

    def pop(self) -> T:
        return heappop(self._heap).value

    def peek(self) -> T:
        return self._heap[0].value

    def __len__(self) -> int:
        return len(self._heap)

    def _itemizer(self, key, reverse):
        op = gt if reverse else lt

        class Item:
            def __init__(self, value):
                self.key, self.value = key(value), value

            def __lt__(self, other: 'Item'):
                return op(self.key, other.key)

        return Item
