import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from ..models.route import Route, Solution
from ..models.problem import Problem

class KNNRouter:
    """
    K-Nearest Neighbors algorithm for routing problems.
    
    This implementation can handle both TSP (Traveling Salesman Problem) and 
    VRP (Vehicle Routing Problem) instances.
    """
    
    def __init__(self, k: int = 5, search_strategy: str = 'greedy', 
                 local_search: bool = True, random_seed: Optional[int] = None):
        """
        Initialize the KNN Router.
        
        Args:
            k: Number of nearest neighbors to consider at each step
            search_strategy: Strategy for selecting the next node ('greedy', 'probabilistic')
            local_search: Whether to apply local search optimization after route construction
            random_seed: Seed for random number generation
        """
        self.k = k
        self.search_strategy = search_strategy
        self.local_search = local_search
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def solve(self, problem: Problem) -> Solution:
        """
        Solve the routing problem using KNN algorithm.
        
        Args:
            problem: The routing problem to solve
            
        Returns:
            A Solution object containing the optimized routes
        """
        # For simplicity, we'll start with TSP
        if problem.problem_type == 'TSP':
            return self._solve_tsp(problem)
        elif problem.problem_type == 'VRP':
            return self._solve_vrp(problem)
        else:
            raise ValueError(f"Unsupported problem type: {problem.problem_type}")

    def _solve_tsp(self, problem: Problem) -> Solution:
        """Solve a TSP instance using KNN."""
        # Get distance matrix and nodes
        dist_matrix = problem.distance_matrix
        num_nodes = dist_matrix.shape[0]
        
        # Start from node 0 (or randomly if specified)
        start_node = 0
        
        # Initialize the route
        route = [start_node]
        unvisited = set(range(num_nodes))
        unvisited.remove(start_node)
        
        # Construct the route using KNN
        current_node = start_node
        while unvisited:
            # Get distances to unvisited nodes
            distances = [(node, dist_matrix[current_node, node]) for node in unvisited]
            
            # Sort by distance and take k nearest
            distances.sort(key=lambda x: x[1])
            candidates = distances[:min(self.k, len(distances))]
            
            # Select next node based on strategy
            if self.search_strategy == 'greedy':
                next_node = candidates[0][0]  # Closest node
            elif self.search_strategy == 'probabilistic':
                # Calculate selection probabilities (inversely proportional to distance)
                total_inv_dist = sum(1/d for _, d in candidates)
                probs = [(1/d)/total_inv_dist for _, d in candidates]
                next_node_idx = np.random.choice(len(candidates), p=probs)
                next_node = candidates[next_node_idx][0]
            else:
                raise ValueError(f"Unsupported search strategy: {self.search_strategy}")
            
            # Add to route and update current node
            route.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
        
        # Close the tour by returning to start
        route.append(start_node)
        
        # Create Route object
        tsp_route = Route(route, problem)
        
        # Apply local search if enabled
        if self.local_search:
            tsp_route = self._apply_local_search(tsp_route, problem)
        
        # Create and return solution
        solution = Solution([tsp_route], problem)
        return solution
    
    def _solve_vrp(self, problem: Problem) -> Solution:
        """Solve a VRP instance using KNN."""
        # Get problem data
        dist_matrix = problem.distance_matrix
        num_nodes = dist_matrix.shape[0]
        depot = problem.depot
        capacities = problem.vehicle_capacities
        demands = problem.demands
        
        # Create routes for each vehicle
        routes = []
        remaining_nodes = set(range(num_nodes))
        remaining_nodes.remove(depot)  # Depot is not a customer
        
        for vehicle_idx, capacity in enumerate(capacities):
            if not remaining_nodes:
                break
                
            # Start route at depot
            route = [depot]
            current_node = depot
            remaining_capacity = capacity
            
            while remaining_nodes:
                # Get feasible nodes (those that fit in remaining capacity)
                feasible_nodes = [n for n in remaining_nodes 
                                if demands[n] <= remaining_capacity]
                
                if not feasible_nodes:
                    break  # No feasible nodes, end this route
                
                # Get distances to feasible nodes
                distances = [(node, dist_matrix[current_node, node]) 
                            for node in feasible_nodes]
                
                # Sort by distance and take k nearest
                distances.sort(key=lambda x: x[1])
                candidates = distances[:min(self.k, len(distances))]
                
                # Select next node (same as TSP)
                if self.search_strategy == 'greedy':
                    next_node = candidates[0][0]
                elif self.search_strategy == 'probabilistic':
                    total_inv_dist = sum(1/d for _, d in candidates)
                    probs = [(1/d)/total_inv_dist for _, d in candidates]
                    next_node_idx = np.random.choice(len(candidates), p=probs)
                    next_node = candidates[next_node_idx][0]
                
                # Add to route and update
                route.append(next_node)
                remaining_nodes.remove(next_node)
                remaining_capacity -= demands[next_node]
                current_node = next_node
            
            # Return to depot
            route.append(depot)
            
            # Create Route object and add to routes
            vrp_route = Route(route, problem, vehicle_idx=vehicle_idx)
            routes.append(vrp_route)
            
            # If all nodes are visited, we're done
            if not remaining_nodes:
                break
        
        # Apply local search to each route if enabled
        if self.local_search:
            routes = [self._apply_local_search(r, problem) for r in routes]
        
        # Create and return solution
        solution = Solution(routes, problem)
        return solution
    
    def _apply_local_search(self, route: Route, problem: Problem) -> Route:
        """Apply 2-opt local search to improve the route."""
        # Implementation of 2-opt local search
        nodes = route.nodes.copy()
        n = len(nodes)
        
        # Don't include the last node if it's the same as the first
        if nodes[0] == nodes[-1]:
            n = n - 1
        
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue  # Skip adjacent edges
                    
                    # Calculate current distance
                    current_dist = (problem.distance_matrix[nodes[i-1], nodes[i]] + 
                                    problem.distance_matrix[nodes[j], nodes[(j+1) % n]])
                    
                    # Calculate new distance after 2-opt swap
                    new_dist = (problem.distance_matrix[nodes[i-1], nodes[j]] + 
                               problem.distance_matrix[nodes[i], nodes[(j+1) % n]])
                    
                    # If improvement found
                    if new_dist < current_dist:
                        # Reverse the segment between i and j
                        nodes[i:j+1] = reversed(nodes[i:j+1])
                        improved = True
                        break
                
                if improved:
                    break
        
        # If it's a closed tour, ensure the last node is the same as the first
        if route.nodes[0] == route.nodes[-1]:
            nodes.append(nodes[0])
        
        # Create a new route with the improved path
        improved_route = Route(nodes, problem, vehicle_idx=route.vehicle_idx)
        return improved_route
