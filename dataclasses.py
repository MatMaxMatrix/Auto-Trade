#%%
from dataclasses import dataclass, field


@dataclass
class InventoryItem:
    '''class for keeping track of an item in inventory'''
    name: str
    unit_price: float
    quantity_on_hand: int = field(default=0)

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
    
item = InventoryItem('widget', 3.0, 10)
print(item.total_cost())
# %%
