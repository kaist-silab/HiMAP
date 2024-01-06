# HiMAP: Learning Heuristics-Informed Policies for Large-Scale Multi-Agent Pathfinding



### Key Files

- `Test_On_DHC_Data_MapSize40.ipynb`: Run the pathfinding on 40\*40 map with obstacle density 0.3; number of agents can be 4, 8, 16, 32, and 64. Output is the average *Success Rate* on all different test cases.
- `Test_On_DHC_Data_MapSize80.ipynb`: Run the pathfinding on 80\*80 map with obstacle density 0.3; number of agents can be 4, 8, 16, 32, and 64. Output is the average *Success Rate* on all different test cases.
- `Test_On_LargeScale_Data_MapSize40.ipynb`: Run the pathfinding on 40\*40 map with obstacle density 0.3; number of agents can be 128, 256, and 512. Output is the average *Success Rate* on all different test cases.
- `Test_On_LargeScale_Data_MapSize80.ipynb`: Run the pathfinding on 80\*80 map with obstacle density 0.3; number of agents can be 128, 256, 512, and 1024. Output is the average *Success Rate* on all different test cases.
- `./data/DHC_Data/`: These test data is taken from [DHC repository](https://github.com/ZiyuanMa/DHC/tree/master/test_set) and is used for tests on DHC data. Each `.pth` file contains 200 different cases given the map specification and the number of agents. 
- `./data/LargeScale_Data/`: Self generated test data used for tests on large-scale data. Each `.pth` file contains 50 different cases given the map specification and the number of agents.

### Running
Click one Jupyter notebook file and run fathfinding with HiMAP on that scenario to see the average success rate. 

