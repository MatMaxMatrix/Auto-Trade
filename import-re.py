import re

text = "The quick brown fox jumps over the lazy dog"
pattern = r"fox"

match = re.search(pattern, text)
if match:
    print("Pattern found:", match.group())

##################################################################
text = "The rain in Spain falls mainly in the plain"
pattern = r"ain"

matches = re.findall(pattern, text)
print("All matches:", matches)

##################################################################

text = "I love apples, but I don't like apple pie"
pattern = r"apple"

new_text = re.sub(pattern, "orange", text)
print("Modified text:", new_text)

##################################################################
text = "The phone number is 123-456-7890"
pattern = r"\d{3}-\d{3}-\d{4}"

match = re.search(pattern, text)
if match:
    print("Phone number found:", match.group())

##################################################################

text = "John Doe's email is john@example.com"
pattern = r"(\w+)@(\w+)\.(\w+)"

match = re.search(pattern, text)
if match:
    print("Full email:", match.group())
    print("Username:", match.group(1))
    print("Domain:", match.group(2))
    print("TLD:", match.group(3))


##################################################################
print("%" * 50)

# Compile the regular expression
action_re = re.compile('^Action: (\w+): (.*)$')

# Example strings
examples = [
    "Action: Jump: over the fence",
    "Action: Run: to the store",
    "Action: Eat: a healthy breakfast",
    "This is not an action",
    "Action: InvalidNoColon"
]

# Test the regular expression
for example in examples:
    match = action_re.match(example)
    if match:
        action_type = match.group(1)
        action_details = match.group(2)
        print(f"Matched: Action Type = '{action_type}', Details = '{action_details}'")
    else:
        print(f"No match: '{example}'")