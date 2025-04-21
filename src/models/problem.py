import numpy as np
from typing import List, Dict, Optional, Union, Any

class Problem:
    """
    Class representing a routing problem instance.
    
    Supports TSP (Traveling Salesman Problem) and VRP (Vehicle Routing Problem).
    """
    
    def __init__(self, 
                 name: str,
                 coordinates: np.ndarray,
                 problem_type: str = 'TSP',
                 depot: int = 0,
                 vehicle_capacities: Optional[List[float]] = None,
                 demands: Optional[List[float]] = None,
                 distance_matrix: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a problem instance.
        
        Args:
            name: Name of the problem
            coordinates: Array of node coordinates (x, y)
            problem_type: Type of problem ('TSP' or 'VRP')
            depot: Index of the depot node (for VRP)
            vehicle_capacities: List of vehicle capacities (for VRP)
            demands: List of customer demands (for VRP)
            distance_matrix: Pre-computed distance matrix (optional)
            metadata: Additional problem information
        """
        self.name = name
        self.coordinates = coordinates
        self.problem_type = problem_type
        self.depot = depot
        self.num_nodes = len(coordinates)
        
        # VRP-specific attributes
        self.vehicle_capacities = vehicle_capacities if vehicle_capacities else []
        self.demands = demands if demands else [0] * self.num_nodes
        
        # Metadata
        self.metadata = metadata if metadata else {}
        
        # Compute distance matrix if not provided
        if distance_matrix is None:
            self.distance_matrix = self._compute_distance_matrix()
        else:
            self.distance_matrix = distance_matrix
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix between all nodes."""
        n = self.num_nodes
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((self.coordinates[i] - self.coordinates[j])**2))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Symmetric
        
        return dist_matrix
    
    def get_distance(self, node1: int, node2: int) -> float:
        """Get the distance between two nodes."""
        return self.distance_matrix[node1, node2]
    
    def get_node_coordinates(self, node: int) -> np.ndarray:
        """Get the coordinates of a node."""
        return self.coordinates[node]
    
    def is_vrp(self) -> bool:
        """Check if this is a VRP problem."""
        return self.problem_type == 'VRP'
    
    def is_tsp(self) -> bool:
        """Check if this is a TSP problem."""
        return self.problem_type == 'TSP'
    
    def __str__(self) -> str:
        """String representation of the problem."""
        return f"{self.problem_type} Problem: {self.name} ({self.num_nodes} nodes)"
