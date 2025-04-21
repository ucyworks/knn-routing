import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.knn import KNNRouter
from src.models.problem import Problem

def generate_dataset(n_nodes_list, n_problems=5, seed=42):
    """Generate benchmark problems of different sizes."""
    np.random.seed(seed)
    
    problems = []
    for n_nodes in n_nodes_list:
        for i in range(n_problems):
            # Generate random coordinates
            coordinates = np.random.rand(n_nodes, 2) * 100
            
            # Create TSP problem
            problem = Problem(
                name=f"Benchmark_N{n_nodes}_{i+1}",
                coordinates=coordinates,
                problem_type='TSP'
            )
            
            problems.append(problem)
    
    return problems

def benchmark_k_values(problems, k_values):
    """Benchmark different k values on the problems."""
    results = []
    
    for problem in tqdm(problems, desc="Problems"):
        for k in k_values:
            # Create router with current k
            router = KNNRouter(k=k, local_search=True)
            
            # Measure solution time
            start_time = time.time()
            solution = router.solve(problem)
            solve_time = time.time() - start_time
            
            # Record results
            results.append({
                'problem_name': problem.name,
                'n_nodes': problem.num_nodes,
                'k': k,
                'distance': solution.total_distance,
                'time_seconds': solve_time,
                'valid': solution.is_valid()
            })
    
    return pd.DataFrame(results)

def plot_results(results_df):
    """Plot benchmark results."""
    # Group by problem size and k value
    grouped = results_df.groupby(['n_nodes', 'k']).agg({
        'distance': 'mean',
        'time_seconds': 'mean'
    }).reset_index()
    
    # Plot distance vs problem size for different k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distance plot
    for k in sorted(grouped['k'].unique()):
        k_data = grouped[grouped['k'] == k]
        ax1.plot(k_data['n_nodes'], k_data['distance'], 'o-', label=f'k={k}')
    
    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Average Tour Distance')
    ax1.set_title('Tour Quality vs Problem Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time plot
    for k in sorted(grouped['k'].unique()):
        k_data = grouped[grouped['k'] == k]
        ax2.plot(k_data['n_nodes'], k_data['time_seconds'], 'o-', label=f'k={k}')
    
    ax2.set_xlabel('Number of Nodes')
    ax2.set_ylabel('Average Solution Time (seconds)')
    ax2.set_title('Computation Time vs Problem Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    # Define problem sizes to benchmark
    n_nodes_list = [10, 20, 50, 100, 200]
    
    # Define k values to test
    k_values = [1, 3, 5, 7, 10]
    
    # Generate benchmark dataset
    print("Generating benchmark problems...")
    problems = generate_dataset(n_nodes_list)
    
    # Run benchmark
    print(f"Benchmarking {len(problems)} problems with {len(k_values)} k values...")
    results = benchmark_k_values(problems, k_values)
    
    # Display summary
    print("\nBenchmark Results Summary:")
    summary = results.groupby(['n_nodes', 'k']).agg({
        'distance': ['mean', 'std'],
        'time_seconds': ['mean', 'std'],
        'valid': 'mean'
    })
    print(summary)
    
    # Plot results
    fig = plot_results(results)
    
    # Save results to CSV
    results.to_csv('benchmark_results.csv', index=False)
    print("Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
