# KNN-Based Routing Optimization

![Routing Optimization](https://img.shields.io/badge/Optimization-Routing-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-KNN-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-orange)

A professional implementation of K-Nearest Neighbors algorithm for solving Vehicle Routing Problems (VRP) and Traveling Salesman Problems (TSP). This repository provides efficient solutions for routing optimization challenges found in academic literature.

## 🚀 Features

- **KNN-Based Route Construction**: Constructs optimized routes using K-Nearest Neighbors algorithm
- **Multiple Problem Support**: Solves TSP, VRP, and their variants
- **Performance Metrics**: Evaluates routes based on total distance, time, and other constraints
- **Visualization Tools**: Generates visual representations of routes and solutions
- **Benchmarks**: Includes comparisons with literature results on standard problem instances

## 📋 Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- NetworkX
- scikit-learn

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/ucytv/knn-routing.git
cd knn-routing

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏁 Quick Start

```python
from src.algorithms.knn import KNNRouter
from src.models.problem import Problem
from src.data_handlers.loader import load_problem

# Load a problem instance
problem = load_problem("data/sample_problems/eil51.tsp")

# Create a KNN router
router = KNNRouter(k=5)

# Solve the problem
solution = router.solve(problem)

# Evaluate the solution
distance = solution.total_distance
print(f"Total distance: {distance}")

# Visualize the solution
solution.visualize()
```

## 📊 Sample Results

| Problem Instance | Nodes | Best Known | KNN Solution | Gap (%) |
|------------------|-------|------------|--------------|---------|
| eil51            | 51    | 426        | 435          | 2.11    |
| berlin52         | 52    | 7542       | 7788         | 3.26    |
| att48            | 48    | 10628      | 10875        | 2.32    |

## 📖 Documentation

### Algorithm Overview

The K-Nearest Neighbors (KNN) approach for routing problems works as follows:

1. **Initialization**: Select a starting node (usually depot in VRP or random in TSP)
2. **Iterative Construction**: 
   - Find K nearest unvisited nodes to the current node
   - Select the best node based on a scoring function
   - Add the selected node to the route
3. **Route Completion**: Connect back to the starting point for TSP or return to depot for VRP
4. **Route Optimization**: Apply local improvement strategies (2-opt, 3-opt) to enhance the solution

### Project Structure

```
knn-routing/
│
├── src/                 # Source code
│   ├── algorithms/      # KNN and other routing algorithms
│   ├── models/          # Problem and solution data models
│   └── data_handlers/   # Data loading and processing utilities
│
├── examples/            # Example usage scripts
│   ├── basic_usage.py   # Basic examples for TSP and VRP
│   └── benchmark.py     # Performance benchmarking
│
├── data/                # Problem instances
│   └── sample_problems/ # Standard test problems
│
└── tests/               # Unit tests
```

## 📚 References

- Laporte, G. (1992). "The vehicle routing problem: An overview of exact and approximate algorithms." European Journal of Operational Research, 59(3), 345-358.
- Lin, S., & Kernighan, B. W. (1973). "An effective heuristic algorithm for the traveling-salesman problem." Operations Research, 21(2), 498-516.
- Toth, P., & Vigo, D. (Eds.). (2002). "The vehicle routing problem." Society for Industrial and Applied Mathematics.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.