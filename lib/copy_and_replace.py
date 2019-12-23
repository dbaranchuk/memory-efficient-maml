""" A convenience function for creating modified copies out-of-place with deepcopy """
from contextlib import contextmanager
from copy import deepcopy

DEFAULT_MEMO = dict()


def copy_and_replace(original, replace=None, do_not_copy=None):
    """
    :param original: object to be copied
    :param replace: a dictionary {old object -> new object}, replace all occurences of old object with new object
    :param do_not_copy: a sequence of objects that will not be copied (but may be replaced)
    :return: a copy of obj with replacements
    """
    replace, do_not_copy = replace or {}, do_not_copy or {}
    memo = dict(DEFAULT_MEMO)
    for item in do_not_copy:
        memo[id(item)] = item

    for item, replacement in replace.items():
        memo[id(item)] = replacement

    return deepcopy(original, memo)


@contextmanager
def do_not_copy(*items):
    """ all calls to copy_and_replace within this context won't copy items (but can replace them) """
    global DEFAULT_MEMO
    keys_to_remove = []
    for item in items:
        key = id(item)
        if key in DEFAULT_MEMO:
            DEFAULT_MEMO[key] = item
            keys_to_remove.append(key)
    
    yield
    
    for key in keys_to_remove:
        DEFAULT_MEMO.pop(key)
