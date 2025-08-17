import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dataset_generation import NetworkTopology


import core_algorithm as _c_ext

@dataclass
class OptimizationResult:
    method: str
    uncertainty_level: str
    x_optimal: np.ndarray
    y_optimal: np.ndarray
    objective_value: float
    total_cost: float
    service_level: float
    computation_time: float
    iterations: int
    convergence_history: Dict[str, List[float]] = field(default_factory=dict)
    cost_per_scenario: np.ndarray = None
    variance: float = 0.0
    unmet_demand: float = 0.0

class BiLevelDRO:
    def __init__(self, network: NetworkTopology, cost_params: dict):
        self.network = network
        self.cost_params = cost_params
        self.n_loc = network.n_locations
    
    def solve(self, demand_scenarios: np.ndarray, epsilon: float,
             max_iterations: int = 20, tolerance: float = 1e-4, error: int = 0) -> OptimizationResult:
        if not isinstance(demand_scenarios, np.ndarray):
            demand_scenarios = np.array(demand_scenarios)
        
        result_dict = _c_ext.solve_dro(demand_scenarios, epsilon, max_iterations, tolerance, error)
        
        return OptimizationResult(
            method=result_dict['method'],
            uncertainty_level=result_dict['uncertainty_level'],
            x_optimal=result_dict['x_optimal'],
            y_optimal=result_dict['y_optimal'],
            objective_value=result_dict['objective_value'],
            total_cost=result_dict['total_cost'],
            service_level=result_dict['service_level'],
            computation_time=result_dict['computation_time'],
            iterations=result_dict['iterations'],
            convergence_history=result_dict.get('convergence_history', {}),
            cost_per_scenario=result_dict['cost_per_scenario'],
            variance=result_dict['variance'],
            unmet_demand=result_dict['unmet_demand']
        )

class RobustOptimization:
    def __init__(self, network: NetworkTopology, cost_params: dict):
        self.network = network
        self.cost_params = cost_params
        self.n_loc = network.n_locations
    
    def solve(self, demand_scenarios: np.ndarray, epsilon: float, error: int = 0) -> OptimizationResult:
        if not isinstance(demand_scenarios, np.ndarray):
            demand_scenarios = np.array(demand_scenarios)
        
        result_dict = _c_ext.solve_ro(demand_scenarios, epsilon, error)
        
        return OptimizationResult(
            method=result_dict['method'],
            uncertainty_level=result_dict['uncertainty_level'],
            x_optimal=result_dict['x_optimal'],
            y_optimal=result_dict['y_optimal'],
            objective_value=result_dict['objective_value'],
            total_cost=result_dict['total_cost'],
            service_level=result_dict['service_level'],
            computation_time=result_dict['computation_time'],
            iterations=result_dict['iterations'],
            convergence_history={},
            cost_per_scenario=result_dict['cost_per_scenario'],
            variance=result_dict['variance'],
            unmet_demand=result_dict['unmet_demand']
        )

class StochasticProgramming:
    def __init__(self, network: NetworkTopology, cost_params: dict):
        self.network = network
        self.cost_params = cost_params
        self.n_loc = network.n_locations
    
    def solve(self, demand_scenarios: np.ndarray, epsilon: float, error: int = 0) -> OptimizationResult:
        if not isinstance(demand_scenarios, np.ndarray):
            demand_scenarios = np.array(demand_scenarios)
        
        result_dict = _c_ext.solve_sp(demand_scenarios, epsilon, error)
        
        return OptimizationResult(
            method=result_dict['method'],
            uncertainty_level=result_dict['uncertainty_level'],
            x_optimal=result_dict['x_optimal'],
            y_optimal=result_dict['y_optimal'],
            objective_value=result_dict['objective_value'],
            total_cost=result_dict['total_cost'],
            service_level=result_dict['service_level'],
            computation_time=result_dict['computation_time'],
            iterations=result_dict['iterations'],
            convergence_history={},
            cost_per_scenario=result_dict['cost_per_scenario'],
            variance=result_dict['variance'],
            unmet_demand=result_dict['unmet_demand']
        )