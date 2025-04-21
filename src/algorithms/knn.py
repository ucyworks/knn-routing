import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from ..models.route import Route, Solution
from ..models.problem import Problem
import time

class KNNRouter:
    """
    K-Nearest Neighbors algorithm for routing problems.
    
    This implementation can handle both TSP (Traveling Salesman Problem) and 
    VRP (Vehicle Routing Problem) instances with enhanced solution quality
    and visualization outputs.
    """
    
    def __init__(self, k: int = 5, search_strategy: str = 'greedy', 
                 local_search: bool = True, random_seed: Optional[int] = None,
                 multi_start: int = 3, time_limit: Optional[float] = None):
        """
        Initialize the KNN Router.
        
        Args:
            k: Number of nearest neighbors to consider at each step
            search_strategy: Strategy for selecting the next node ('greedy', 'probabilistic')
            local_search: Whether to apply local search optimization after route construction
            random_seed: Seed for random number generation
            multi_start: Number of different starting points to try
            time_limit: Maximum running time in seconds (None for unlimited)
        """
        self.k = k
        self.search_strategy = search_strategy
        self.local_search = local_search
        self.multi_start = multi_start
        self.time_limit = time_limit
        self.start_time = None
        
        # Set random seed if provided
        self.random_seed = random_seed
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
        self.start_time = time.time()
        
        # For TSP, try multiple starting nodes and pick the best solution
        if problem.problem_type == 'TSP':
            return self._solve_tsp_with_multi_start(problem)
        elif problem.problem_type == 'VRP':
            return self._solve_vrp_with_validation(problem)
        else:
            raise ValueError(f"Unsupported problem type: {problem.problem_type}")
    
    def _is_time_limit_reached(self) -> bool:
        """Check if the time limit has been reached."""
        if self.time_limit is None:
            return False
        return time.time() - self.start_time > self.time_limit
    
    def _solve_tsp_with_multi_start(self, problem: Problem) -> Solution:
        """Solve TSP with multiple starting points and pick the best solution."""
        best_solution = None
        best_distance = float('inf')
        
        # Try different starting nodes
        start_nodes = [0]  # Always include node 0
        if self.multi_start > 1:
            # Add some random nodes as starting points
            additional_nodes = np.random.choice(
                range(1, problem.num_nodes), 
                min(self.multi_start - 1, problem.num_nodes - 1), 
                replace=False
            )
            start_nodes.extend(additional_nodes)
        
        for start_node in start_nodes:
            if self._is_time_limit_reached():
                break
                
            # Solve with current starting node
            solution = self._solve_tsp(problem, start_node)
            
            # Keep track of the best solution
            if solution.total_distance < best_distance:
                best_solution = solution
                best_distance = solution.total_distance
        
        return best_solution
    
    def _solve_tsp(self, problem: Problem, start_node: int = 0) -> Solution:
        """Solve a TSP instance using KNN starting from the specified node."""
        # Get distance matrix and nodes
        dist_matrix = problem.distance_matrix
        num_nodes = dist_matrix.shape[0]
        
        # Initialize the route
        route = [start_node]
        unvisited = set(range(num_nodes))
        unvisited.remove(start_node)
        
        # Construct the route using KNN
        current_node = start_node
        while unvisited:
            if self._is_time_limit_reached():
                # If time limit reached, connect remaining nodes greedily
                while unvisited:
                    # Get closest unvisited node
                    next_node = min(unvisited, key=lambda n: dist_matrix[current_node, n])
                    route.append(next_node)
                    unvisited.remove(next_node)
                    current_node = next_node
                break
                
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
                total_inv_dist = sum(1/max(0.0001, d) for _, d in candidates)  # Avoid division by zero
                probs = [(1/max(0.0001, d))/total_inv_dist for _, d in candidates]
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
        if self.local_search and not self._is_time_limit_reached():
            tsp_route = self._apply_local_search(tsp_route, problem)
        
        # Create and return solution
        solution = Solution([tsp_route], problem)
        return solution
    
    def _solve_vrp_with_validation(self, problem: Problem) -> Solution:
        """Solve VRP with additional validation to ensure feasible solutions."""
        # First try the regular VRP solver
        solution = self._solve_vrp(problem)
        
        # Validate the solution - check if all nodes are visited
        all_visited = set()
        for route in solution.routes:
            for node in route.nodes:
                if node != problem.depot:
                    all_visited.add(node)
        
        all_nodes = set(range(problem.num_nodes))
        all_nodes.remove(problem.depot)
        
        unvisited = all_nodes - all_visited
        
        # If there are unvisited nodes, try to insert them into existing routes
        if unvisited and not self._is_time_limit_reached():
            solution = self._insert_unvisited_nodes(solution, problem, unvisited)
        
        return solution
    
    def _insert_unvisited_nodes(self, solution: Solution, problem: Problem, unvisited: set) -> Solution:
        """Insert unvisited nodes into the solution using a greedy insertion approach."""
        routes = solution.routes.copy()
        
        # For each unvisited node
        for node in unvisited:
            if self._is_time_limit_reached():
                break
                
            best_insertion = None
            best_cost = float('inf')
            best_route_idx = -1
            
            # Find the best insertion point across all routes
            for i, route in enumerate(routes):
                # Skip if node demand exceeds route capacity
                if problem.is_vrp():
                    remaining_capacity = problem.vehicle_capacities[route.vehicle_idx] - route.load
                    if problem.demands[node] > remaining_capacity:
                        continue
                
                # Find best insertion point in this route
                nodes = route.nodes
                for j in range(1, len(nodes)):  # Skip inserting at depot
                    if nodes[j] == problem.depot and j < len(nodes) - 1:
                        continue  # Don't insert between depot visits
                        
                    # Calculate insertion cost
                    prev_node = nodes[j-1]
                    next_node = nodes[j]
                    
                    # Cost increase calculation
                    old_cost = problem.get_distance(prev_node, next_node)
                    new_cost = (problem.get_distance(prev_node, node) + 
                                problem.get_distance(node, next_node))
                    delta = new_cost - old_cost
                    
                    if delta < best_cost:
                        best_cost = delta
                        best_insertion = j
                        best_route_idx = i
            
            # Insert the node at the best position
            if best_insertion is not None:
                # Update the route
                new_nodes = routes[best_route_idx].nodes.copy()
                new_nodes.insert(best_insertion, node)
                
                # Create a new route with the inserted node
                new_route = Route(
                    new_nodes, 
                    problem, 
                    vehicle_idx=routes[best_route_idx].vehicle_idx
                )
                
                # Apply local search to the modified route
                if self.local_search and not self._is_time_limit_reached():
                    new_route = self._apply_local_search(new_route, problem)
                
                routes[best_route_idx] = new_route
            else:
                # If insertion fails, create a new route if possible
                if problem.is_vrp() and len(routes) < len(problem.vehicle_capacities):
                    new_vehicle_idx = len(routes)
                    new_route = Route(
                        [problem.depot, node, problem.depot], 
                        problem, 
                        vehicle_idx=new_vehicle_idx
                    )
                    routes.append(new_route)
        
        # Create a new solution with the updated routes
        return Solution(routes, problem)
    
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
        """Apply enhanced 2-opt local search to improve the route."""
        # Implementation of 2-opt local search with time limit check
        nodes = route.nodes.copy()
        n = len(nodes)
        
        # Don't include the last node if it's the same as the first
        is_loop = False
        if nodes[0] == nodes[-1]:
            n = n - 1
            is_loop = True
        
        # Improvements counter for reporting
        improvement_count = 0
        
        improved = True
        max_iterations = min(100, n*n)  # Limit iterations for large problems
        iterations = 0
        
        while improved and iterations < max_iterations:
            if self._is_time_limit_reached():
                break
                
            improved = False
            iterations += 1
            
            # Try all possible 2-opt swaps
            for i in range(1, n - 1):
                if improved or self._is_time_limit_reached():
                    break
                    
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue  # Skip adjacent edges
                    
                    # Calculate current distance
                    current_dist = (problem.get_distance(nodes[i-1], nodes[i]) + 
                                   problem.get_distance(nodes[j], nodes[(j+1) % n]))
                    
                    # Calculate new distance after 2-opt swap
                    new_dist = (problem.get_distance(nodes[i-1], nodes[j]) + 
                               problem.get_distance(nodes[i], nodes[(j+1) % n]))
                    
                    # If improvement found
                    if new_dist < current_dist:
                        # Reverse the segment between i and j
                        nodes[i:j+1] = reversed(nodes[i:j+1])
                        improved = True
                        improvement_count += 1
                        break
        
        # If it's a closed tour, ensure the last node is the same as the first
        if is_loop:
            nodes.append(nodes[0])
        
        # Create a new route with the improved path
        improved_route = Route(nodes, problem, vehicle_idx=route.vehicle_idx)
        return improved_route
