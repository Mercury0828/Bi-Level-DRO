import numpy as np
import pandas as pd
from typing import Dict, List

from core_algorithm_wrap import OptimizationResult

class ResultsAnalyzer:
    
    @staticmethod
    def create_summary_table(results: Dict[str, OptimizationResult]) -> pd.DataFrame:
        
        summary_data = []
        for key, result in results.items():
            summary_data.append({
                'Method': result.method,
                'Uncertainty': result.uncertainty_level,
                'Total Cost': f"{result.total_cost:.2f}",
                'Service Level (%)': f"{result.service_level * 100:.2f}",
            })
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def compute_cost_reduction(results: Dict[str, OptimizationResult],
                              baseline_cost: float) -> Dict[str, float]:
        
        reductions = {}
        for key, result in results.items():
            reduction = (1 - result.total_cost / baseline_cost) * 100
            reductions[key] = reduction
        
        return reductions
    
    @staticmethod
    def analyze_robustness(results: Dict[str, OptimizationResult]) -> Dict:
        
        robustness = {}
        
        methods = {}
        for key, result in results.items():
            method = result.method
            if method not in methods:
                methods[method] = []
            methods[method].append(result)
        
        for method, method_results in methods.items():
            costs = [r.total_cost for r in method_results]
            services = [r.service_level for r in method_results]
            
            robustness[method] = {
                'cost_variance': np.var(costs),
                'cost_range': max(costs) - min(costs),
                'service_variance': np.var(services),
                'avg_service': np.mean(services)
            }
        
        return robustness

