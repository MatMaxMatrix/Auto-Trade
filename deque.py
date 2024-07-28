#%%
from collections import deque

# Create a new deque
fruit_basket = deque(['apple', 'banana', 'orange'])

# Add an item to the right end
fruit_basket.append('grape')

# Add an item to the left end
fruit_basket.appendleft('pear')
print("Fruit basket:", fruit_basket)
# Remove and return an item from the right end
last_fruit = fruit_basket.pop()
print("Fruit basket:", fruit_basket)
# Remove and return an item from the left end
first_fruit = fruit_basket.popleft()

print("Fruit basket:", fruit_basket)
print("Last fruit removed:", last_fruit)
print("First fruit removed:", first_fruit)

# %%
class math():
    def __init__(self, x: int, y):
        self.x = x
        self.y = y
    def add(self):
        return self.x + self.y
class math2(math):
    def __init__(self, x, y):
        super().__init__(x, y)
    def add(self):
        base_add = super().add()
        return base_add + 1
m = math2(3, 2)
print(m.add())
# %%

import logging 
logging.basicConfig(level=logging.INFO)
logging.info('Hello, world!')

# %%
import logging

logging.basicConfig(level=logging.CRITICAL)

logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')

#%%
import logging

def divide_numbers(x, y):
    try:
        result = x / y
    except ZeroDivisionError as e:
        logging.error("Tried to divide by zero")
        return None
    else:
        return result

logging.basicConfig(level=logging.INFO)
print(divide_numbers(1, 0))# %%

# %%
import logging

def log_messages():
    logging.debug('This is a debug message')
    logging.info('This is an info message')
    logging.warning('This is a warning message')
    logging.error('This issss an error message')
    logging.critical('Thissss is a critical message')

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)
print("^^^^^^^^^^^")
print("Logging level set to DEBUG")
log_messages()
print("%%%%%%%%%%%")
# Set logging level to INFO
logging.basicConfig(level=logging.INFO)
print("\nLogging level set to INFO")
log_messages()

# Set logging level to WARNING
logging.basicConfig(level=logging.WARNING)
print("\nLogging level set to WARNING")
log_messages()

# Set logging level to ERROR
logging.basicConfig(level=logging.ERROR)
print("\nLogging level set to ERROR")
log_messages()

# Set logging level to CRITICAL
logging.basicConfig(level=logging.CRITICAL)
print("\nLogging level set to CRITICAL")
log_messages()
 # %%
