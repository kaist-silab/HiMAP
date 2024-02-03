# HiMAP: Learning Heuristics-Informed Policies for Large-Scale Multi-Agent Pathfinding

![image](https://github.com/kaist-silab/HiMAP/blob/main/assets/HiMAP_Overview.jpg)

### Key Files

- `test.py`: Run pathfinding to see the *Success Rate*.
- `config.py`: Parameters for test data generation, etc.
- `datagen.py`: Test data generator.
- `./data/DHC_Data/`: These test data is taken from [DHC repository](https://github.com/ZiyuanMa/DHC/tree/master/test_set) and is used for tests on DHC data. Each `.pth` file contains 200 different cases given the map specification and the number of agents. 
- `./data/LargeScale_Data/`: Self generated test data used for tests on large-scale data. Each `.pth` file contains 50 different cases given the map specification and the number of agents.
