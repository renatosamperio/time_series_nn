
def reduce_options(options, sorted = True):
    reduced = list(set(options))
    if sorted:
        reduced.sort()
    return reduced