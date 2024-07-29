#%%
# Let's create a simple text file
with open('example.txt', 'w') as file:
    file.write("Hello, World! This is a seek example.")

# Now, let's read from this file
with open('example.txt', 'r') as file:
    # Initially, the file pointer is at the beginning (position 0)
    print(file.read(5))  # Outputs: Hello

    # Let's use seek to move the file pointer
    file.seek(7)  # Move to the 8th character (remember, it's 0-indexed)
    print(file.read(5))  # Outputs: World

    # Let's move to the end of "This"
    file.seek(14)
    print(file.read(2))  # Outputs: is

    # We can also move backwards
    file.seek(0)  # Back to the start
    print(file.read(5))  # Outputs: Hello again
# %%
