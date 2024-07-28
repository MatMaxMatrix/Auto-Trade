#Simple example to show the Callable
from typing import Callable


def math_operation(x: int, y: int, operation: Callable[[int, int], int]) -> int:
	return operation(x, y)

def add(x: int, y: int) -> int:
	return x + y

def mul(x:int, y:int) -> int:
	return x*y

print(math_operation(2,3,add))
print(math_operation(2,3,mul))
