from io import StringIO

string_array = ['Hello', ' ', 'world', '!']
s = StringIO()
for str in string_array:
    s.write(str)
result = s.getvalue()
print(result)  # Output: Hello world!