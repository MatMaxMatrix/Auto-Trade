
"""
import argparse
my_parser = argparse.ArgumentParser(description = "This is a simple argument parser")

my_parser.add_argument('FirstArgument', metavar = 'first', type = int , help='an integer for the first argument')
my_parser.add_argument('SecondArgument', metavar = 'second', type = str , help='an integer for the second argument')

args = my_parser.parse_args()
print(f"First argument: {args.FirstArgument} and the second argument is {args.SecondArgument}")
"""
import argparse
class ArgParseExample():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description = "This is a simple argument parser")
        self.add_arguments()
    def add_arguments(self):
        self.parser.add_argument('FirstArg', metavar='first_argument', type=int, help='an integer as the first argument')
        self.parser.add_argument('SecondArg', metavar='second_argument', type=str, help='a string as the second argument')
        self.parser.add_argument('--optional', metavar='optional_argument', type=str, help='an optional string argument')
    def run(self):
        args = self.parser.parse_args()
        print(f"The first argument is {args.FirstArg}, the second argument is {args.SecondArg}, and the optional argument is {args.optional}")
if __name__ == "__main__":
    ArgParseExample().run()