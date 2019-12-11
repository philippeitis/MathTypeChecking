import builtins

real_import = builtins.__import__

available_types = {}


def dynamic_type_importer(name, global_vars, local_vars, fromlist, level):
    if name == "MagicTypes":
        if "MagicTypes" not in globals():
            global MagicTypes
            MagicTypes = real_import("MagicTypes", global_vars, local_vars, [], level)
            global meta_parser
            meta_parser = MagicTypes.meta_parser

        to_return = []
        for import_str in fromlist:
            if import_str not in dir(MagicTypes):
                result = meta_parser(import_str)
                to_return.append(result)
                available_types[import_str] = result
                setattr(MagicTypes, import_str, result)
    return real_import(name, global_vars, local_vars, fromlist, level)


builtins.__import__ = dynamic_type_importer
if __name__ == "__main__":
    from MagicTypes import int_on10_10o

    print(int_on10_10o)

    from MagicTypes import float_on10_10o

    print(float_on10_10o)

    num = float("5e-7")
    print(num)
    print(num == int(num))
    from MagicTypes import int_on10e1_10o
    print(int_on10e1_10o)
