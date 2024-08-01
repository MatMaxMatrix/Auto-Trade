#%%
#DICTIONARY

from typing import Dict 
def get_name_and_age(user: Dict[str, int]) -> str:
    return f"{user['name']} is {user['age']} years old"
get_name_and_age({"name": "John", "age": 30})
# %%
#LIST

from typing import List
def get_first_element(elements: List[int]) -> int:
    return elements[0]
get_first_element([1, 2, 3])
# %%

#TUPLE
from typing import Tuple 
def get_coordinates(coordinates: Tuple[int, int]) -> str:
    return f"Latitude: {coordinates[0]}, Longitude: {coordinates[1]}"
get_coordinates((37, -122))
# %%

#SET
from typing import Set
def get_unique_elements(elements: Set[int]) -> List[int]:
    return list(elements)
get_unique_elements({1, 2, 3, 3, 3})
# %%

#SEQUENCE   
from typing import Sequence
def get_last_element(elements: Sequence[int]) -> int:
    return elements[-1]
get_last_element([1, 2, 3])
# %%

#ITERABLE
from typing import Iterable
def get_elements(elements: Iterable[int]) -> List[int]:
    return list(elements)
get_elements({1, 2, 3})
# %%

#MAPPING
from typing import Mapping
def get_values(mapping: Mapping[str, int]) -> List[int]:
    return list(mapping.values())
get_values({"a": 1, "b": 2, "c": 3})
# %%

#KEYS_VIEW
from typing import KeysView
def get_keys(mapping: KeysView[str]) -> List[str]:
    return list(mapping)
get_keys({"a": 1, "b": 2, "c": 3}.keys())
# %%

#VALUES_VIEW
from typing import ValuesView
def get_values(mapping: ValuesView[int]) -> List[int]:
    return list(mapping)
get_values({"a": 1, "b": 2, "c": 3}.values())
# %%

#ITEMS_VIEW
from typing import ItemsView
def get_items(mapping: ItemsView[str, int]) -> List[Tuple[str, int]]:
    return list(mapping)
get_items({"a": 1, "b": 2, "c": 3}.items())
# %%


#CHAINMAP
from typing import ChainMap
def get_values(mapping: ChainMap[str, int]) -> List[int]:
    return list(mapping.values())
get_values(ChainMap({"a": 1}, {"b": 2}, {"c": 3}))
# %%

#TYPE
from typing import Type
def get_type(obj: Type[int]) -> str:
    return str(obj)
get_type(int)
# %%

#TYPEVAR
from typing import TypeVar
T = TypeVar("T")    
def get_type(obj: T) -> str:
    return str(obj)
get_type(1)
# %%

#UNION
from typing import Union  
def get_type(obj: Union[int, str]) -> str:
    return str(obj)
get_type(1)

# %%

#OPTIONAL
from typing import Optional
def get_type(obj: Optional[int]) -> str:
    return str(obj)
get_type(1)
# %%

#ANY
from typing import Any
def get_type(obj: Any) -> str:
    return str(obj)
get_type(1)
# %%

#NO_RETURN
from typing import NoReturn
def raise_error() -> NoReturn:
    raise ValueError("An error occurred")
raise_error()
# %%

#NEW_TYPE
from typing import NewType
UserId = NewType("UserId", int)
def get_user_id(user_id: UserId) -> str:
    return str(user_id)
get_user_id(UserId(1))
# %%

#CLASS
from typing import ClassVar
class MyClass:
    class_var: ClassVar[int] = 10
    def __init__(self, instance_var: int) -> None:
        self.instance_var = instance_var
obj = MyClass(20)
obj.class_var

# %%

#TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mymodule import MyClass
def get_class_var(obj: MyClass) -> int:
    return obj.class_var

# %%
