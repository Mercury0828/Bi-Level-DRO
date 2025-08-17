import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class NetworkTopology:
    n_locations: int
    n_routes: int
    distance_matrix: np.ndarray
    storage_capacity: np.ndarray
    transport_capacity: np.ndarray
    
class DatasetGenerator:
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_network_topology(self, n_locations: int = 10, 
                                 n_routes: int = 15) -> NetworkTopology:
        
        # Generate location coordinates
        coords = np.random.rand(n_locations, 2) * 100
        
        # Calculate distance matrix
        distance_matrix = np.zeros((n_locations, n_locations))
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        # Generate capacities
        storage_capacity = np.random.uniform(80, 120, n_locations)
        transport_capacity = np.random.uniform(40, 60, (n_locations, n_locations))
        
        return NetworkTopology(
            n_locations=n_locations,
            n_routes=n_routes,
            distance_matrix=distance_matrix,
            storage_capacity=storage_capacity,
            transport_capacity=transport_capacity
        )
    
    def generate_demand_scenarios(self, n_locations: int, n_scenarios: int,
                                 mean_demand: float = 50.0, std_demand: float = 15.0,
                                 uncertainty_level: float = 0.1) -> np.ndarray:
        
        # Base demand distribution (gamma distribution as per paper)
        base_demand = np.random.gamma(
            shape=2.0,
            scale=mean_demand / 2.0,
            size=n_locations
        )
        
        scenarios = []
        for _ in range(n_scenarios):
            # Add perturbation within Wasserstein ball
            perturbation = np.random.normal(
                0,
                uncertainty_level * std_demand,
                n_locations
            )
            
            # Ensure non-negative demand
            scenario = np.maximum(base_demand + perturbation, 0)
            scenarios.append(scenario)
        
        return np.array(scenarios)
    
    def generate_forecast_error_scenarios(self, base_demand: np.ndarray,
                                         forecast_errors: list,
                                         n_scenarios: int = 50) -> dict:
        
        error_scenarios = {}
        for error in forecast_errors:
            perturbed_scenarios = []
            for _ in range(n_scenarios):
                perturbation = base_demand * (error / 100.0)
                noise = np.random.normal(0, 5, len(base_demand))
                scenario = np.maximum(base_demand + perturbation + noise, 0)
                perturbed_scenarios.append(scenario)
            error_scenarios[error] = np.array(perturbed_scenarios)
        
        return error_scenarios

