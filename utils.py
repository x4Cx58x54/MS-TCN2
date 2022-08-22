def read_nonempty_lines(path):
    with open(path, 'r') as f:
        lines = [line for line in map(lambda x: x.strip(), f.readlines()) if line]
    return lines
