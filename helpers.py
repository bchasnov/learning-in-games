import ast

def add_argument(parser, desc, default, help_str, 
                 metavar=None, arg_type=None, choices=None, action=None):
    """ adds argument to argparse.ArugmentParser """
    desc = desc.replace("_", '-')
    if arg_type is None:
        arg_type = type(default)
    elif type(arg_type) is list:
        choices = arg_type
        arg_type = type(arg_type[0])
    elif type(arg_type) is bool:
        arg_type = None
        action = 'store_true'
    parser.add_argument('--'+desc, type=arg_type, default=default, metavar=metavar,
            help=help_str, choices=choices)
    return parser

def load_dict(filename):
    with open(filename) as f:
        kwargs = ast.literal_eval(f.readline())
    return kwargs

