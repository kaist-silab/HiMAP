def load_map_data(file):
    """Load map data from `.map` file"""
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    map_data = lines[4:]
    map_data = "\n".join(map_data)
    return map_data
