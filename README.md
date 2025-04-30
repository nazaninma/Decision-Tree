# Decision-Tree
# Decision Tree Classifier Project

## Overview  
This project implements a multi-node categorical decision tree classifier from scratch in Python. The decision tree can be built using either Gini index or entropy as the splitting criterion.

## Features  
- **Multi-node splits**  
- **Two splitting criteria**:  
  - Gini index (default)  
  - Entropy (information gain)  
- **Tree pruning controls**:  
  - `max_depth`  
  - `min_samples_split`  
- Feature importance calculation  
- Scikit-learn compatible API  

## Requirements  
- Python 3.6+  
- NumPy  
- scikit-learn  

## Installation  
```bash
git clone https://github.com/yourusername/decision-tree-project.git
cd decision-tree-project
pip install -r requirements.txt
