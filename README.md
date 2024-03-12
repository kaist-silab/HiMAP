# HiMAP: Learning Heuristics-Informed Policies for Large-Scale Multi-Agent Pathfinding

[![arXiv](https://img.shields.io/badge/arXiv-2402.15546-b31b1b.svg)](https://arxiv.org/abs/2402.15546)


![image](https://github.com/kaist-silab/HiMAP/blob/main/assets/HiMAP_Overview.jpg)

### Key Files

- `test.py`: Run pathfinding to see the *Success Rate*.
- `config.py`: Parameters for test data generation, etc.
- `datagen.py`: Test data generator.
- `./data/DHC_Data/`: These test data is taken from [DHC repository](https://github.com/ZiyuanMa/DHC/tree/master/test_set) and is used for tests on DHC data. Each `.pth` file contains 200 different cases given the map specification and the number of agents. 
- `./data/LargeScale_Data/`: Self generated test data used for tests on large-scale data. Each `.pth` file contains 50 different cases given the map specification and the number of agents.

### Citation

If you find our code or work helpful, please consider citing us:
```bibtex
@inproceedings{tang2024himap,
  title={Hi{MAP}: Learning Heuristics-Informed Policies for Large-Scale Multi-Agent Pathfinding},
  author={Tang, Huijie and Berto, Federico and Ma, Zihan and Hua, Chuanbo and Ahn, Kyuree and Park, Jinkyoo},
  booktitle={AAMAS},
  year={2024}
}
```
