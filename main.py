
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

from dataset_generation import DatasetGenerator, NetworkTopology
from core_algorithm_wrap import (
    BiLevelDRO, 
    RobustOptimization, 
    StochasticProgramming,
    OptimizationResult
)
from visualization import PaperVisualizer
from utils import ResultsAnalyzer


def run_experiments():
    
    print("=" * 60)
    print("Bi-Level DRO Experiments - Paper Implementation")
    print("=" * 60)
    
    data_gen = DatasetGenerator(seed=42)
    
    network = data_gen.generate_network_topology(n_locations=10, n_routes=15)
    
    cost_params = {
        'inventory_cost': 5.0,
        'transport_cost': 10.0,
        'penalty_cost': 20.0
    }
    
    uncertainty_levels = {
        'Low': 0.05,
        'Medium': 0.10,
        'High': 0.20
    }
    
    dro_solver = BiLevelDRO(network, cost_params)
    ro_solver = RobustOptimization(network, cost_params)
    sp_solver = StochasticProgramming(network, cost_params)
    
    results = {}
    
    for unc_name, epsilon in uncertainty_levels.items():
        print(f"\n--- Uncertainty Level: {unc_name} (ε={epsilon:.2f}) ---")
        
        # Generate demand scenarios
        demand_scenarios = data_gen.generate_demand_scenarios(
            n_locations=network.n_locations,
            n_scenarios=100,
            uncertainty_level=epsilon
        )
        
        print("  Running DRO...", end=" ")
        dro_result = dro_solver.solve(demand_scenarios, epsilon)
        results[f'DRO_{unc_name}'] = dro_result
        print(f"Cost: {dro_result.total_cost:.2f}, Service: {dro_result.service_level:.2%}")
        
        print("  Running SP...", end=" ")
        sp_result = sp_solver.solve(demand_scenarios, epsilon)
        results[f'SP_{unc_name}'] = sp_result
        print(f"Cost: {sp_result.total_cost:.2f}, Service: {sp_result.service_level:.2%}")
        
        print("  Running RO...", end=" ")
        ro_result = ro_solver.solve(demand_scenarios, epsilon)
        results[f'RO_{unc_name}'] = ro_result
        print(f"Cost: {ro_result.total_cost:.2f}, Service: {ro_result.service_level:.2%}")
    
    print("\n" + "=" * 60)
    print("Running Forecast Error Analysis")
    print("=" * 60)
    
    forecast_errors = [-10, 0, 10, 20]
    forecast_results = {}
    
    base_demand = data_gen.generate_demand_scenarios(
        n_locations=network.n_locations,
        n_scenarios=1,
        uncertainty_level=0.11
    )[0]
    
    for error in forecast_errors:
        print(f"  Forecast error: {error}%")
        error_scenarios = data_gen.generate_forecast_error_scenarios(
            base_demand, [error], n_scenarios=50
        )[error]
        forecast_results[error] = {
            'DRO': dro_solver.solve(error_scenarios, 0.11, error=error),
            'SP': sp_solver.solve(error_scenarios, 0.11, error=error),
            'RO': ro_solver.solve(error_scenarios, 0.11, error=error)
        }
    
    print("\n" + "=" * 60)
    print("Running Scalability Analysis")
    print("=" * 60)
    
    problem_sizes = [10, 20, 50, 80]
    scalability_results = {}
    
    for size in problem_sizes:
        print(f"  Problem size: {size}")
        
        scaled_network = data_gen.generate_network_topology(n_locations=size)
        
        scaled_dro = BiLevelDRO(scaled_network, cost_params)
        scaled_sp = StochasticProgramming(scaled_network, cost_params)
        scaled_ro = RobustOptimization(scaled_network, cost_params)
        
        scaled_scenarios = data_gen.generate_demand_scenarios(
            n_locations=size,
            n_scenarios=50,
            uncertainty_level=0.1
        )
        
        scalability_results[size] = {
            'DRO': scaled_dro.solve(scaled_scenarios, 0.1),
            'SP': scaled_sp.solve(scaled_scenarios, 0.1),
            'RO': scaled_ro.solve(scaled_scenarios, 0.1)
        }
    
    print("\n" + "=" * 60)
    print("Generating Paper Figures")
    print("=" * 60)
    
    visualizer = PaperVisualizer(results)
    
    fig2 = visualizer.plot_figure2_cost_service()
    fig3 = visualizer.plot_figure4_cost_distribution()
    fig4 = visualizer.plot_figure5_inventory_variance(forecast_results)
    fig5 = visualizer.plot_figure6_computational_time(scalability_results)
    
    fig2.savefig('../visualization/figure2_cost_service.png', dpi=300, bbox_inches='tight')
    fig3.savefig('../visualization/figure4_cost_distribution.png', dpi=300, bbox_inches='tight')
    fig4.savefig('../visualization/figure5_inventory_variance.png', dpi=300, bbox_inches='tight')
    fig5.savefig('../visualization/figure6_computational_time.png', dpi=300, bbox_inches='tight')
    
    print("  ✓ All figures saved")
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    analyzer = ResultsAnalyzer()
    summary_table = analyzer.create_summary_table(results)
    print("\n", summary_table.to_string(index=False))

    plt.show()
    
    return results, forecast_results, scalability_results

if __name__ == "__main__":
    results, forecast_results, scalability_results = run_experiments()