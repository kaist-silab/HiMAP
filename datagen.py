# Generate test data
import subprocess
from pathlib import Path
import numpy as np
import pickle
import shutil
import config

PATH_TO_EECBS = config.PATH_TO_EECBS
MAX_ITERATIONS = config.MAX_ITERATIONS

def flood_fill(matrix, x, y, old_value, new_value):
    """Performs the flood fill algorithm"""
    if x < 0 or x >= matrix.shape[0] or y < 0 or y >= matrix.shape[1]:
        return
    if matrix[x, y] != old_value:
        return
    matrix[x, y] = new_value
    flood_fill(matrix, x+1, y, old_value, new_value)
    flood_fill(matrix, x-1, y, old_value, new_value)
    flood_fill(matrix, x, y+1, old_value, new_value)
    flood_fill(matrix, x, y-1, old_value, new_value)

def generate_map(width, height, density, tolerance=0.005):
    """Generates a grid map with specified width, height, and obstacle density"""
    
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        matrix = np.random.choice([0, 1], size=(width, height), p=[1-density, density])
        
        # Clone the matrix to keep the original for calculation purposes
        filled_matrix = matrix.copy()
        
        # Use flood fill from top-left to mark all reachable cells with value 2
        flood_fill(filled_matrix, 0, 0, 0, 2)
        
        # Calculate the reachable free space
        total_free_space = np.sum(filled_matrix == 2)
        total_space = width * height
        actual_density = 1 - total_free_space/total_space
        
        # If the actual density is close to desired density, finalize the matrix
        if abs(actual_density - density) < tolerance:
            # After flood fill, change all 0 (unreachable free cells) to 1 (obstacles)
            filled_matrix[filled_matrix == 0] = 1
            
            # Change the 2's back to 0's
            filled_matrix[filled_matrix == 2] = 0
            
            return filled_matrix
        
        iteration += 1
    
    # If we couldn't achieve the desired density in max_iterations
    raise ValueError(f"Unable to generate a grid with the desired density of {density} after {MAX_ITERATIONS} iterations.")

def save_env_map_to_file(map, file_path):
    with open(file_path, 'w') as file:
        file.write("type octile\n")
        file.write(f"height {map.shape[0]}\n")
        file.write(f"width {map.shape[1]}\n")
        file.write("map\n")
        for row in map:
            line = ''.join(['.' if cell == 0 else '@' for cell in row]) + '\n'
            file.write(line)

def generate_instance(dim, density, num_instances):
    folder = Path(f'./temp/dim{dim}_density{density}')
    folder.mkdir(parents=True, exist_ok=True) 
    for i in range(num_instances):
        generator = generate_map(dim, dim, density)
        env_map_file_path = folder / f'random_{dim}_{density}_case_{i+1}.map'
        save_env_map_to_file(generator, env_map_file_path)

def run_eecbs_on_files(base_dir='./temp', k=4, t=15, suboptimality=10):
 
    # Assuming the base_dir is 'instances' or any other directory containing your generated folders
    folders = [folder for folder in Path(base_dir).iterdir() if folder.is_dir()]

    for folder in folders:
        
        map_files = list(folder.glob('*.map'))
        # scen_files = list(folder.glob('*.scen'))

        for map_file in map_files:
            output_base = f'{map_file.stem}'
            cmd = [
                PATH_TO_EECBS,
                '-m', str(map_file),
                '-a', f'{folder}/{output_base}_agents_{k}.scen',
                # '-o', f'{folder}/{output_base}_agents_{k}_stats.csv',           
                # '--outputPaths', f'{folder}/{output_base}_agents_{k}_paths.txt', 
                '-k', str(k),
                '-t', str(t),
                '--suboptimality', str(suboptimality)
            ]
            subprocess.run(cmd)

def process_and_save_to_pickle(map_filepaths, start_goal_filepaths, output_filename):
    # Ensure the number of map files matches the number of start/goal files
    assert len(map_filepaths) == len(start_goal_filepaths), "Mismatch in number of map files and start/goal files."
    
    data = []
    
    for map_filepath, start_goal_filepath in zip(map_filepaths, start_goal_filepaths):
        # Read the map file
        with open(map_filepath, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        # Read the start/goal file
        with open(start_goal_filepath, 'r') as f:
            start_goal_content = f.read()

        # Convert map content to numpy array
        map_data = lines[4:]
        map_size = len(map_data)
        map_array = np.zeros((map_size, map_size), dtype=int)
        
        for i, line in enumerate(map_data):
            for j, char in enumerate(line):
                if char == '@':
                    map_array[i, j] = 1  # obstacle
        

        # Convert start/goal content to numpy arrays
        start_goal_lines = start_goal_content.split("\n")[1:-1]  # skip the first line and last empty line
        start_locations = np.array([list(map(int, line.split(",")[:2])) for line in start_goal_lines])
        goal_locations = np.array([list(map(int, line.split(",")[2:4])) for line in start_goal_lines])
        
        data.append((map_array, start_locations, goal_locations))
    
    # Save the data to a pickle file
    with open(output_filename, 'wb') as f:
        pickle.dump(data, f)

def main(dim=80, density=0.3, num_agents=4, num_instances=100):
    '''
    dim: length of the map
    density: density of the obstacles in the map
    num_agents: number of agents in the map
    num_instances: number of cases (problems) in one .pth file
    '''
    generate_instance(dim=dim, density=density, num_instances=num_instances)

    run_eecbs_on_files('./temp', num_agents ,t=0.01)

    # map_filepaths = list(Path(f'./temp/dim{dim}_density{density}').glob('*.map'))   #NOTE: FATAL ERROR: ORDER MATTERS, IF GET ALL MAPS IN THIS WAY, ORDER IS RANDOM. 
    # start_goal_filepaths = list(Path(f'./temp/dim{dim}_density{density}').glob('*.scen'))  # This will cause the map and start/goal pairs to be mismatched.

    map_filepaths = [f"./temp/dim{dim}_density{density}/random_{dim}_{density}_case_{i+1}.map" for i in range(num_instances)]
    start_goal_filepaths = [f"./temp/dim{dim}_density{density}/random_{dim}_{density}_case_{i+1}_agents_{num_agents}.scen" for i in range(num_instances)]

    output_filename=f"./data/{dim}length_{num_agents}agents_{density}density.pth"

    process_and_save_to_pickle(map_filepaths, start_goal_filepaths, output_filename)


# NOTE: Generated instances should be solvable by EECBS, otherwise the problem itself may be impossible to solve.(Invalid Instance)
if __name__ == '__main__':
    for num_agents in config.NUM_AGENTS:
        main(config.MAP_SIZE, config.MAP_DENSITY, num_agents, config.NUM_CASES)
        # Remove the temp folder, REMOVE IS NECESSARY BEFORE NEXT RUNNNING OF MAIN FUNCTION
        shutil.rmtree('./temp')