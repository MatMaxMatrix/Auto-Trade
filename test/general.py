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
