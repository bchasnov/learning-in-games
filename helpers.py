import ast

def load_dict(filename):
    with open(filename) as f:
        kwargs = ast.literal_eval(f.readline())
    return kwargs