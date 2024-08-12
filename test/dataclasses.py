#%%
from dataclasses import dataclass, field

@dataclass
class InventoryItem:
    '''class for keeping track of an item in inventory'''
    name: str = field(default='myItem')
    unit_price: float = field(default=0.0)
    quantity_on_hand: int = field(default=0)

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
    
item = InventoryItem('widget', 3.0, 10)
print(item.total_cost())
# %%
#from the second class, we can inherit from the first class
@dataclass
class child(InventoryItem):
    pass

test = child()
print(child.name)
print(child.unit_price)


# %%
# in the second class, the second class will initialized the first class
@dataclass
class InventoryItem2():
    '''class for keeping track of an item in inventory'''
    name: str = field(default='myItem')
    unit_price: float = field(default=0.0)
    quantity_on_hand: int = field(default=0)

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
    
class child2(InventoryItem2):
    def __init__(self, name):
        super().__init__(name=name)

item = child2("Iphone")
print(item.name)


    # %%
