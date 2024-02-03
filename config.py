# Git clone EECBS repository from https://github.com/Jiaoyang-Li/EECBS/tree/main
# Write the path to the EECBS executable in the PATH_TO_EECBS variable
PATH_TO_EECBS = '/home/huijie/Downloads/MAPF_Repo/Heuristic/EECBS-main/eecbs'

# Parameters for generating test set
MAX_ITERATIONS = 1000 # Maximum number of iterations to try to generate a map with desired density
MAP_DENSITY = 0.3 
MAP_SIZE = 40 
NUM_AGENTS = [4, 8, 16, 32, 64] 
NUM_CASES = 50

# Parameters for testing
SOFTMAX_TEMPERATURE = 2.0 # was 1.0
HISTORY_SIZE = 5
ngpu = 1
model_path = 'checkpoints/vggnet_params_best33_dataloader1.pth'

# Parameters for for testing on DHC dataset
NUM_TEST_CASES = 200
TEST_DATA_PATH = './data/DHC_Data/'

# Parameters for testing on large scale dataset
# NUM_TEST_CASES = 50
# TEST_DATA_PATH = './data/LargeScale_Data/'

# General testing parameters
# Assume the map size is 40
TEST_MAP_SIZE = 40 # or 80 if map size is 80
TEST_MAX_TIMESTEP = 256 # or 386 if map size is 80

NUM_TEST_AGENTS = [4, 8, 16, 32, 64] # DHC dataset
# NUM_TEST_AGENTS = [128, 256, 512] # LargeScale dataset for 40x40 map
# NUM_TEST_AGENTS = [128, 256, 512, 1024] # LargeScale dataset for 80x80 map
