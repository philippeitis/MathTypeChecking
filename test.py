from MagicTypes import meta_parser
import builtins

real_import = builtins.__import__

available_types = {}


def dynamic_type_importer(name, global_vars, local_vars, fromlist, level):
    if name == "MagicTypes":
        to_return = []
        for import_str in fromlist:
            result = meta_parser(import_str)
            to_return.append(result)
            available_types[import_str] = result
            print(result)
        return to_return
    return real_import(name, global_vars, local_vars, fromlist, level)


builtins.__import__ = dynamic_type_importer
if __name__ == "__main__":
    from MagicTypes import int_on10_10o
    print(int_on10_10o)

