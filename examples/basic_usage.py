import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.knn import KNNRouter
from src.models.problem import Problem
from src.data_handlers.loader import load_problem

def generate_random_tsp(n_nodes=20, seed=42):
    """Generate a random TSP problem."""
    np.random.seed(seed)
    coordinates = np.random.rand(n_nodes, 2) * 100
    return Problem(
        name=f"Random{n_nodes}",
        coordinates=coordinates,
        problem_type='TSP'
    )

def generate_random_vrp(n_nodes=20, n_vehicles=3, seed=42):
    """Generate a random VRP problem."""
    np.random.seed(seed)
    coordinates = np.random.rand(n_nodes, 2) * 100
    
    # Set the first node as depot
    depot = 0
    
    # Generate random demands for each node (except depot)
    demands = [0] + list(np.random.randint(1, 20, size=n_nodes-1))
    
    # Vehicle capacities (all the same for simplicity)
    vehicle_capacities = [100] * n_vehicles
    
    return Problem(
        name=f"RandomVRP{n_nodes}",
        coordinates=coordinates,
        problem_type='VRP',
        depot=depot,
        vehicle_capacities=vehicle_capacities,
        demands=demands
    )

def solve_and_visualize(problem, k=5):
    """Solve a problem using KNN and visualize the solution."""
    # Create KNN router
    router = KNNRouter(k=k, local_search=True)
    
    # Solve the problem
    print(f"Solving {problem.name}...")
    solution = router.solve(problem)
    
    # Print solution info
    print(solution)
    
    # Visualize solution
    solution.visualize()
    
    return solution

def main():
    # Example 1: Solve a random TSP
    print("Example 1: Solving a random TSP")
    tsp_problem = generate_random_tsp(n_nodes=30)
    tsp_solution = solve_and_visualize(tsp_problem, k=3)
    
    # Example 2: Solve a random VRP
    print("\nExample 2: Solving a random VRP")
    vrp_problem = generate_random_vrp(n_nodes=25, n_vehicles=3)
    vrp_solution = solve_and_visualize(vrp_problem, k=3)

if __name__ == "__main__":
    main()
