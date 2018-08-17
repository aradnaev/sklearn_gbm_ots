def dict_to_lists(d, target_type=list):
    return map(target_type, zip(*(d.items())))
