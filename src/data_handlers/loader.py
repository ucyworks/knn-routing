import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import re
from ..models.problem import Problem

def load_problem(filepath: str) -> Problem:
    """
    Load a routing problem from a file.
    
    Supports various formats:
    - TSPLIB (.tsp)
    - CSV coordinates
    - Custom VRP format
    
    Args:
        filepath: Path to the problem file
        
    Returns:
        A Problem instance
    """
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.tsp':
        return _load_tsplib(filepath)
    elif file_ext == '.csv':
        return _load_csv(filepath)
    elif file_ext == '.vrp':
        return _load_vrp(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def _load_tsplib(filepath: str) -> Problem:
    """Load a problem in TSPLIB format."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse the header
    name = ""
    dimension = 0
    edge_weight_type = ""
    node_coord_section = False
    coords = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            node_coord_section = True
            coord_start_line = i + 1
            break
    
    # Read node coordinates
    if node_coord_section:
        for i in range(coord_start_line, coord_start_line + dimension):
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 3:  # node_id, x, y
                    coords.append((float(parts[1]), float(parts[2])))
    
    # Convert to numpy array
    coordinates = np.array(coords)
    
    # Create and return the problem
    return Problem(
        name=name,
        coordinates=coordinates,
        problem_type='TSP'
    )

def _load_csv(filepath: str) -> Problem:
    """Load a problem from a CSV file containing node coordinates."""
    try:
        data = pd.read_csv(filepath)
        
        # Get problem name from filename
        name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Extract coordinates
        if 'x' in data.columns and 'y' in data.columns:
            coordinates = data[['x', 'y']].values
        else:
            # If x,y not specified, use first two columns
            coordinates = data.iloc[:, :2].values
        
        # Create and return the problem (assume TSP by default)
        return Problem(
            name=name,
            coordinates=coordinates,
            problem_type='TSP'
        )
    
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")

def _load_vrp(filepath: str) -> Problem:
    """Load a VRP problem from a custom VRP format."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse the file
    name = ""
    dimension = 0
    capacity = 0
    depot = 0
    node_coord_section = False
    demand_section = False
    
    coords = []
    demands = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line.startswith("CAPACITY"):
            capacity = float(line.split(":")[1].strip())
        elif line.startswith("DEPOT_SECTION"):
            i += 1
            if i < len(lines):
                depot = int(lines[i].strip()) - 1  # Convert to 0-indexed
        elif line.startswith("NODE_COORD_SECTION"):
            node_coord_section = True
            i += 1
            
            # Read coordinates
            for _ in range(dimension):
                if i < len(lines):
                    parts = lines[i].strip().split()
                    if len(parts) >= 3:  # node_id, x, y
                        coords.append((float(parts[1]), float(parts[2])))
                    i += 1
                
            node_coord_section = False
            continue
        
        elif line.startswith("DEMAND_SECTION"):
            demand_section = True
            i += 1
            
            # Read demands
            for _ in range(dimension):
                if i < len(lines):
                    parts = lines[i].strip().split()
                    if len(parts) >= 2:  # node_id, demand
                        demands.append(float(parts[1]))
                    i += 1
                
            demand_section = False
            continue
        
        i += 1
    
    # Ensure demand array is the right size
    if len(demands) < dimension:
        demands.extend([0] * (dimension - len(demands)))
    
    # Convert to numpy array
    coordinates = np.array(coords)
    
    # Create problem with single vehicle
    return Problem(
        name=name,
        coordinates=coordinates,
        problem_type='VRP',
        depot=depot,
        vehicle_capacities=[capacity],
        demands=demands
    )
