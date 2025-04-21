from typing import List, Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from .problem import Problem

class Route:
    """Class representing a route in a routing problem."""
    
    def __init__(self, nodes: List[int], problem: Problem, vehicle_idx: Optional[int] = None):
        """
        Initialize a route.
        
        Args:
            nodes: List of node indices defining the route
            problem: The problem this route belongs to
            vehicle_idx: Vehicle index (for VRP problems)
        """
        self.nodes = nodes
        self.problem = problem
        self.vehicle_idx = vehicle_idx
        
        # Calculate distance
        self.distance = self._calculate_distance()
        
        # For VRP, calculate load
        if problem.is_vrp():
            self.load = self._calculate_load()
    
    def _calculate_distance(self) -> float:
        """Calculate the total distance of the route."""
        total_distance = 0.0
        for i in range(len(self.nodes) - 1):
            total_distance += self.problem.get_distance(self.nodes[i], self.nodes[i+1])
        return total_distance
    
    def _calculate_load(self) -> float:
        """Calculate the total load of the route (for VRP)."""
        if not self.problem.is_vrp():
            return 0.0
        
        total_load = 0.0
        for node in self.nodes:
            if node != self.problem.depot:  # Don't count depot
                total_load += self.problem.demands[node]
        return total_load
    
    def is_valid(self) -> bool:
        """Check if the route is valid."""
        # For VRP, check capacity constraint
        if self.problem.is_vrp():
            if self.vehicle_idx is None or self.vehicle_idx >= len(self.problem.vehicle_capacities):
                return False
            
            if self.load > self.problem.vehicle_capacities[self.vehicle_idx]:
                return False
        
        # Check if the route starts and ends at the depot for VRP
        # or forms a complete tour for TSP
        if self.problem.is_vrp():
            return self.nodes[0] == self.problem.depot and self.nodes[-1] == self.problem.depot
        else:  # TSP
            return self.nodes[0] == self.nodes[-1]
    
    def visualize(self, ax=None, show: bool = False):
        """Visualize the route."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            show_plot = True
        else:
            show_plot = show
        
        # Extract coordinates
        coords = np.array([self.problem.get_node_coordinates(node) for node in self.nodes])
        
        # Plot nodes
        ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, alpha=0.7)
        
        # Highlight depot
        if self.problem.is_vrp():
            depot_coord = self.problem.get_node_coordinates(self.problem.depot)
            ax.scatter([depot_coord[0]], [depot_coord[1]], c='red', s=100, marker='*')
        
        # Plot route
        ax.plot(coords[:, 0], coords[:, 1], 'k-', alpha=0.5)
        
        # Add node labels
        for i, node in enumerate(self.nodes):
            ax.annotate(str(node), (coords[i, 0], coords[i, 1]), 
                         xytext=(5, 5), textcoords='offset points')
        
        # Set title
        vehicle_info = f" (Vehicle {self.vehicle_idx})" if self.vehicle_idx is not None else ""
        title = f"Route{vehicle_info}: Distance = {self.distance:.2f}"
        if self.problem.is_vrp():
            title += f", Load = {self.load:.2f}"
        ax.set_title(title)
        
        if show_plot:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    def __str__(self) -> str:
        """String representation of the route."""
        vehicle_info = f" (Vehicle {self.vehicle_idx})" if self.vehicle_idx is not None else ""
        return f"Route{vehicle_info}: {' -> '.join(map(str, self.nodes))}, Distance: {self.distance:.2f}"


class Solution:
    """Class representing a solution to a routing problem."""
    
    def __init__(self, routes: List[Route], problem: Problem):
        """
        Initialize a solution.
        
        Args:
            routes: List of routes comprising the solution
            problem: The problem this solution belongs to
        """
        self.routes = routes
        self.problem = problem
        
        # Calculate total distance
        self.total_distance = sum(route.distance for route in routes)
        
        # For VRP, calculate total load
        if problem.is_vrp():
            self.total_load = sum(route.load for route in routes)
    
    def is_valid(self) -> bool:
        """Check if the solution is valid."""
        # Check if all routes are valid
        if not all(route.is_valid() for route in self.routes):
            return False
        
        # For TSP, we should have exactly one route
        if self.problem.is_tsp() and len(self.routes) != 1:
            return False
        
        # Check if all nodes are visited exactly once
        visited_nodes = set()
        
        for route in self.routes:
            for node in route.nodes:
                # Skip depot for VRP
                if self.problem.is_vrp() and node == self.problem.depot:
                    continue
                    
                # For TSP, skip the last node which is the same as the first
                if self.problem.is_tsp() and node == route.nodes[-1] and node == route.nodes[0]:
                    continue
                
                if node in visited_nodes:
                    return False
                visited_nodes.add(node)
        
        # Check if all non-depot nodes are visited
        all_nodes = set(range(self.problem.num_nodes))
        if self.problem.is_vrp():
            all_nodes.remove(self.problem.depot)
        
        return visited_nodes == all_nodes
    
    def visualize(self):
        """Visualize the entire solution."""
        if len(self.routes) == 1:
            # For TSP or single-route VRP
            fig, ax = plt.subplots(figsize=(10, 8))
            self.routes[0].visualize(ax)
        else:
            # For multi-route VRP
            num_routes = len(self.routes)
            fig, axes = plt.subplots(nrows=(num_routes+1)//2, ncols=2, 
                                    figsize=(15, 5*((num_routes+1)//2)))
            
            # Handle case with just 2 routes
            if num_routes == 2:
                for i, route in enumerate(self.routes):
                    route.visualize(axes[i])
            else:
                # Flatten if multiple rows
                if num_routes > 2:
                    axes = axes.flatten()
                
                for i, route in enumerate(self.routes):
                    route.visualize(axes[i])
                
                # Hide unused subplots
                for i in range(num_routes, len(axes)):
                    axes[i].axis('off')
        
        plt.suptitle(f"{self.problem.name} - Total Distance: {self.total_distance:.2f}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    def __str__(self) -> str:
        """String representation of the solution."""
        result = [f"Solution for {self.problem.name} - Total Distance: {self.total_distance:.2f}"]
        for i, route in enumerate(self.routes):
            result.append(f"  {route}")
        return "\n".join(result)
