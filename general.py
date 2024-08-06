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


#re.findall:This function searches a string for all occurrences of a pattern and returns them as a list of strings.

questions_text = """1. What is the capital of France?

2. Who wrote "Romeo and Juliet"?

3. What is the chemical symbol for gold?

4. What year did World War II end?"""

import re

questions = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', questions_text, re.DOTALL) #\d+\.: Matches 1., 2., 3.' and \s*: Matches any spaces after the period. and (.*?): Captures the question text. AND (?=\n\d+\.|\Z): Ensures we stop capturing at the next question or the end of the text.
print(questions)
# %%
