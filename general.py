#%%
class MyClass:
    def __init__(self):
        self.my_attribute = "Hello, world!"

# Create an instance of MyClass
my_instance = MyClass()

# Check if 'my_attribute' exists in my_instance
print(hasattr(my_instance, 'my_attribute'))  # Outputs: True

# Check if 'non_existent_attribute' exists in my_instance
print(hasattr(my_instance, 'non_existent_attribute'))  # Outputs: False
# %%
class MyClass:
    def __init__(self):
        self.my_attribute = "Hello, world!"

    def my_method(self):
        return "Hello from my_method!"

# Create an instance of MyClass
my_instance = MyClass()

# Check if 'my_attribute' exists in my_instance
print(hasattr(my_instance, 'my_attribute'))  # Outputs: True

# Check if 'my_method' exists in my_instance
print(hasattr(my_instance, 'my_method'))  # Outputs: True

# Check if 'non_existent_attribute' exists in my_instance
print(hasattr(my_instance, 'non_existent_attribute'))  # Outputs: False






#%%
import copy

# A list of lists
original = [[1], [2], [3]]

# Create a shallow copy
shallow_copy = copy.copy(original)

# Create a deep copy
deep_copy = copy.deepcopy(original)

# Modify the original list
original[0][0] = 'a'

print(original)        # Outputs: [['a'], [2], [3]]
print(shallow_copy)    # Outputs: [['a'], [2], [3]]
print(deep_copy)       # Outputs: [[1], [2], [3]]
# %%
