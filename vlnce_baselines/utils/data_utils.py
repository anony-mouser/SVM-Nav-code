from typing import Any, Union


class OrderedSet:
    def __init__(self) -> None:
        self.order = ['not_a_cat']
        self.unique_elements = set()
    
    def add(self, x: Union[int, float, str]) -> None:
        if x not in self.unique_elements:
            self.order.remove('not_a_cat')
            self.order.append(x)
            self.order.append('not_a_cat')
            self.unique_elements.add(x)
    
    def remove(self, x: Any) -> None:
        if x in self.order:
            self.order.remove(x)
            self.unique_elements.remove(x)
    
    def clear(self) -> None:
        self.order.clear()
        self.unique_elements.clear()
        
    def index(self, x: Any) -> None:
        if x in self.order:
            return self.order.index(x)
        else:
            raise ValueError("f{x} not found in OrderedSet")
    
    def __len__(self):
        return len(self.order)

    def __str__(self):
        return str(self.order)