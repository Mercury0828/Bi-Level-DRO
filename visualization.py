import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import matplotlib 

from core_algorithm_wrap import OptimizationResult

class PaperVisualizer:
    
    def __init__(self, results: Dict[str, OptimizationResult]):
        self.results = results
        self._setup_style()
    
    def _setup_style(self):
        """Set paper style"""
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--'
        })
    
    def plot_figure2_cost_service(self) -> plt.Figure:
        """Figure 2: Cost and service levels from actual results"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        uncertainties = ['Low', 'Medium', 'High']
        x_pos = np.arange(len(uncertainties))
        width = 0.25
        
        dro_costs = []
        sp_costs = []
        ro_costs = []
        dro_service = []
        sp_service = []
        ro_service = []
        
        for unc in uncertainties:
            if f'DRO_{unc}' in self.results:
                dro_costs.append(self.results[f'DRO_{unc}'].total_cost)
                dro_service.append(self.results[f'DRO_{unc}'].service_level * 100)
            if f'SP_{unc}' in self.results:
                sp_costs.append(self.results[f'SP_{unc}'].total_cost)
                sp_service.append(self.results[f'SP_{unc}'].service_level * 100)
            if f'RO_{unc}' in self.results:
                ro_costs.append(self.results[f'RO_{unc}'].total_cost)
                ro_service.append(self.results[f'RO_{unc}'].service_level * 100)
        
        bars1 = ax.bar(x_pos - width, dro_costs, width, label='DRO (Costs)',
                       color='none', edgecolor='black', linewidth=1.5, hatch='/')
        bars2 = ax.bar(x_pos, sp_costs, width, label='SP (Costs)',
                       color='none', edgecolor='black', linewidth=1.5, hatch='\\')
        bars3 = ax.bar(x_pos + width, ro_costs, width, label='RO (Costs)',
                       color='none', edgecolor='black', linewidth=1.5, hatch='-')
        
        ax2 = ax.twinx()
        ax2.plot(uncertainties, dro_service, 'o-', color='orange',
                linewidth=2, markersize=8, label='DRO (Service)')
        ax2.plot(uncertainties, sp_service, 'o--', color='yellow',
                linewidth=2, markersize=8, label='SP (Service)')
        ax2.plot(uncertainties, ro_service, 'o-.', color='red',
                linewidth=2, markersize=8, label='RO (Service)')
        
        ax.set_xlabel('Uncertainty Level', fontsize=16)
        ax.set_ylabel('Total Logistics Costs (Thousands)', fontsize=16)
        ax2.set_ylabel('Service Level (%)', fontsize=16)
        ax.set_title('Total Logistics Costs and Service Levels Across Uncertainty Levels',
                    fontsize=18)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(uncertainties)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=14)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    def plot_figure4_cost_distribution(self) -> plt.Figure:
        """Figure 4: Cost distribution from actual results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        box_data = []
        
        if 'DRO_Medium' in self.results:
            box_data.append(self.results['DRO_Medium'].cost_per_scenario)
        if 'SP_Medium' in self.results:
            box_data.append(self.results['SP_Medium'].cost_per_scenario)
        if 'RO_Medium' in self.results:
            box_data.append(self.results['RO_Medium'].cost_per_scenario)
        
        if box_data:
            positions = [1, 2, 3][:len(box_data)]
            
            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=False, showmeans=False,
                           medianprops=dict(color='orange', linewidth=2),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            labels = ['DRO', 'SP', 'RO'][:len(box_data)]
            ax.set_xticklabels(labels)
        
        ax.set_xlabel('Model', fontsize=16)
        ax.set_ylabel('Total Cost (Thousands)', fontsize=16)
        ax.set_title('Total Cost Distribution Across Models', fontsize=18)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_figure5_inventory_variance(self, forecast_results: Dict) -> plt.Figure:
        """Figure 5: Inventory variance and unmet demand"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        forecast_errors = sorted(forecast_results.keys())
        x_pos = np.arange(len(forecast_errors))
        width = 0.25
        
        dro_variance = []
        sp_variance = []
        ro_variance = []
        dro_unmet = []
        sp_unmet = []
        ro_unmet = []
        
        for error in forecast_errors:
            if error in forecast_results:
                methods_data = forecast_results[error]
                if 'DRO' in methods_data:
                    dro_variance.append(methods_data['DRO'].variance)
                    dro_unmet.append(methods_data['DRO'].unmet_demand)
                if 'SP' in methods_data:
                    sp_variance.append(methods_data['SP'].variance)
                    sp_unmet.append(methods_data['SP'].unmet_demand)
                if 'RO' in methods_data:
                    ro_variance.append(methods_data['RO'].variance)
                    ro_unmet.append(methods_data['RO'].unmet_demand)
        
        if dro_variance:
            ax.bar(x_pos - width, dro_variance, width,
                   label='DRO Variance', color='none',
                   edgecolor='black', linewidth=1.5, hatch='/')
        if sp_variance:
            ax.bar(x_pos, sp_variance, width,
                   label='SP Variance', color='none',
                   edgecolor='black', linewidth=1.5, hatch='\\')
        if ro_variance:
            ax.bar(x_pos + width, ro_variance, width,
                   label='RO Variance', color='none',
                   edgecolor='black', linewidth=1.5, hatch='-')
        
        ax2 = ax.twinx()
        if dro_unmet:
            ax2.plot(x_pos, dro_unmet, 'o-', color='orange',
                    linewidth=2, markersize=8, label='DRO Unmet Demand')
        if sp_unmet:
            ax2.plot(x_pos, sp_unmet, 'o--', color='darkorange',
                    linewidth=2, markersize=8, label='SP Unmet Demand')
        if ro_unmet:
            ax2.plot(x_pos, ro_unmet, 'o-.', color='red',
                    linewidth=2, markersize=8, label='RO Unmet Demand')
        
        ax.set_xlabel('Forecast Error (%)', fontsize=16)
        ax.set_ylabel('Inventory Variance', fontsize=16)
        ax2.set_ylabel('Unmet Demand (%)', fontsize=16)
        ax.set_title('Inventory Variance and Unmet Demand vs. Forecast Error', fontsize=18)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(forecast_errors)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',fontsize=14)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig
    
    def plot_figure6_computational_time(self, scalability_results: Dict) -> plt.Figure:
        """Figure 6: Computational scalability from actual results"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        problem_sizes = sorted(scalability_results.keys())
        x_pos = np.arange(len(problem_sizes))
        width = 0.25
        
        dro_quality = []
        sp_quality = []
        ro_quality = []
        dro_time = []
        sp_time = []
        ro_time = []
        
        for size in problem_sizes:
            if size in scalability_results:
                size_data = scalability_results[size]
                if 'DRO' in size_data:
                    dro_quality.append(size_data['DRO'].service_level)
                    dro_time.append(size_data['DRO'].computation_time)
                if 'SP' in size_data:
                    sp_quality.append(size_data['SP'].service_level)
                    sp_time.append(size_data['SP'].computation_time)
                if 'RO' in size_data:
                    ro_quality.append(size_data['RO'].service_level)
                    ro_time.append(size_data['RO'].computation_time)
        
        if dro_quality:
            ax.bar(x_pos - width, dro_quality, width,
                   label='DRO Quality', color='none',
                   edgecolor='black', linewidth=1.5, hatch='/')
        if sp_quality:
            ax.bar(x_pos, sp_quality, width,
                   label='SP Quality', color='none',
                   edgecolor='black', linewidth=1.5, hatch='\\')
        if ro_quality:
            ax.bar(x_pos + width, ro_quality, width,
                   label='RO Quality', color='none',
                   edgecolor='black', linewidth=1.5, hatch='-')
        
        ax2 = ax.twinx()
        if dro_time:
            ax2.plot(x_pos, dro_time, 'o-', color='orange',
                    linewidth=2, markersize=8, label='DRO Time')
        if sp_time:
            ax2.plot(x_pos, sp_time, 'o--', color='darkorange',
                    linewidth=2, markersize=8, label='SP Time')
        if ro_time:
            ax2.plot(x_pos, ro_time, 'o-.', color='red',
                    linewidth=2, markersize=8, label='RO Time')
        
        ax.set_xlabel('Problem Size', fontsize=16)
        ax.set_ylabel('Solution Quality (Normalized)', fontsize=16)
        ax2.set_ylabel('Computational Time (Seconds)', fontsize=16)
        ax.set_title('Solution Quality and Computational Time vs. Problem Size', fontsize=18)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(problem_sizes)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=14)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig

