""" from MagicTypes import [integral type]_[o|c]FIRSTNUM_SECONDNUM[o|c]
    numbers can either be arbitrary if the integral type has its own parser: eg. any string not containing a _,
    or they can be structured for the program: d stands in for the dot, n stangs in for negatives, so an open interval
    over -10, 10 would look like int_on10_10o.
    To escape characters, use E: magicE_type_0n10_10c
    If magic_type has it's own parser, it should be available in registered_types and parsers.
    """


# import builtins

# real_import = builtins.__import__
#
# available_types = []
#
#
# def dynamic_type_importer(name, global_vars, local_vars, fromlist, level):
#     if name == "MagicTypes":
#         to_return = {}
#         for import_str in fromlist:
#             result = meta_parser(import_str)
#             to_return.append(result)
#             available_types[import_str] = result
#         return result
#     return real_import(name, global_vars, local_vars, fromlist, level)
#
#
# builtins.__import__ = dynamic_type_importer


def int_parser(num_as_str):
    num_as_str = num_as_str.replace("n", "-")
    num = float(num_as_str)
    int_num = int(num)
    if num != int_num:
        raise ValueError(f"Called int constructor on a non-int number: {num_as_str} becomes {num}")
    return int(float(num_as_str))


def float_parser(num_as_str):
    num_as_str = num_as_str.replace("n", "-")
    num_as_str = num_as_str.replace("d", ".")
    return float(num_as_str)


def float_interval_parser(interval_str):
    # Receives the entire string after int_
    from MathAtomic import IntervalObject
    left_open = interval_str[0] == "o"
    right_open = interval_str[-1] == "o"
    split_str = interval_str.split("_")
    return IntervalObject(float_parser(split_str[0][1:]), left_open, float_parser(split_str[1][:-1]), right_open)


def int_interval_parser(interval_str):
    # Receives the entire string after int_
    from MathAtomic import IntervalObject
    left_open = interval_str[0] == "o"
    right_open = interval_str[-1] == "o"
    split_str = interval_str.split("_")
    return IntervalObject(int_parser(split_str[0][1:]), left_open, int_parser(split_str[1][:-1]), right_open)


def default_interval_parser(interval_str, constructor):
    from MathAtomic import IntervalObject
    left_open = interval_str[0] == "o"
    right_open = interval_str[-1] == "o"
    split_str = interval_str.split("_")
    return IntervalObject(constructor(split_str[0][1:]), left_open, constructor(split_str[1][:-1]), right_open)

# Any types you plan on using should be in this array
registered_types = [int, float]

# And should have corresponding parsers linked here, or linked in this file as [type_name]_parser.
# These should either return None on parsing error or raise an actual error.
parsers = {"int": int_parser, "float": float_parser}

# Alternatively, you could specify the whole interval string yourself - if you do this, make it available below.
interval_parsers = {"int": int_interval_parser}


# Or you could even roll your own string parser, in which case, you should specify that as this parser
def meta_parser(import_str: str):
    first_split = import_str.index("_")
    interval_type = import_str[:first_split]
    interval_str = import_str[first_split + 1:]
    if interval_type not in interval_parsers:
        return default_interval_parser(interval_str, parsers[interval_type])
    return interval_parsers.get(interval_type)(interval_str)
