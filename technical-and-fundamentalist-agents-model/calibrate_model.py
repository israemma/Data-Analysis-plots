"""
Bayesian optimization for financial agent-based model parameter calibration.

Optimizes market parameters to match stylized facts including kurtosis,
skewness, geometric distribution fit, and VReturns statistics.
"""

import os
import numpy as np
import pandas as pd
import subprocess
import shutil
import json
import time
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

from experiment_multi import run_experiment_repetitions
from market_model import generate_market_data


class ModelCalibrator:
    """Calibrate financial ABM parameters using Bayesian optimization."""

    def __init__(self, target_kurtosis=6.5, minimum_kurtosis=3.0, target_skew=0.0, min_geom_pvalue=0.05, target_vreturns_mean=0.009):
        """Initialize calibrator with target stylized facts.
        
        Args:
            target_kurtosis (float): Target kurtosis value.
            minimum_kurtosis (float): Minimum acceptable kurtosis.
            target_skew (float): Target skewness value.
            min_geom_pvalue (float): Minimum p-value for geometric fit.
            target_vreturns_mean (float): Target mean for VReturns.
        """
        self.target_kurtosis = target_kurtosis
        self.minimum_kurtosis = minimum_kurtosis
        self.target_skew = target_skew
        self.min_geom_pvalue = min_geom_pvalue
        self.target_vreturns_mean = target_vreturns_mean
        
        self.num_steps = 3000
        self.num_repetitions = 20
        self.trial_folders = []
        
    def run_simulation_and_analyze(self, price_impact, noise_std, fundamental_proportion, 
                                 reversion_theta, fundamental_sigma, threshold_low, threshold_high, num_agents):
        """Execute simulation and calculate fitness score.
        
        Args:
            price_impact (float): Price impact parameter.
            noise_std (float): Noise standard deviation.
            fundamental_proportion (float): Proportion of fundamental agents.
            reversion_theta (float): Mean reversion speed.
            fundamental_sigma (float): Fundamental volatility.
            threshold_low (float): Lower threshold for fundamental agents.
            threshold_high (float): Upper threshold for fundamental agents.
            num_agents (int): Number of agents in simulation.
        
        Returns:
            float: Fitness score (higher is better).
        """
        # Track trial iteration
        if not hasattr(self, 'current_trial_count'):
            self.current_trial_count = 0
        self.current_trial_count += 1
        trial_id = f"trial_{self.current_trial_count}"
        
        try:
            num_agents = int(round(num_agents))
            
            # Prepare market parameters
            market_params = {
                'price_impact': price_impact,
                'noise_std': noise_std,
                'fundamental_proportion': fundamental_proportion,
                'reversion_theta': reversion_theta,
                'fundamental_sigma': fundamental_sigma,
                'fundamental_threshold_range': (threshold_low, threshold_high)
            }

            print(f"\n--- Trial {trial_id} ---")
            print(f"Parameters: {market_params}")
            
            # Create experiment folder
            main_iterations_folder = 'experiment iterations'
            if not os.path.exists(main_iterations_folder):
                os.makedirs(main_iterations_folder)
            
            iteration_folder_name = f'iteration_{trial_id}'
            permanent_folder = os.path.join(main_iterations_folder, iteration_folder_name)
            
            if os.path.exists(permanent_folder):
                shutil.rmtree(permanent_folder)
            
            # Run experiment
            start_time = time.time()
            run_experiment_repetitions(
                experiment_idx=1,
                params=market_params,
                experiments_root_folder=permanent_folder,
                num_agents=num_agents,
                num_steps=self.num_steps,
                num_repetitions=self.num_repetitions
            )
            
            # Execute analysis scripts
            print(f"Running analysis scripts for {permanent_folder}...")
            
            try:
                result = subprocess.run(['python', 'plots_multiple_repetitions.py', permanent_folder], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"Warning: plots_multiple_repetitions.py failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("Warning: plots_multiple_repetitions.py timed out")
            except Exception as e:
                print(f"Warning: Error running plots_multiple_repetitions.py: {e}")
            
            try:
                result = subprocess.run(['python', 'trend_acceleration_plots_pooled.py', permanent_folder], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"Warning: trend_acceleration_plots_pooled.py failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("Warning: trend_acceleration_plots_pooled.py timed out")
            except Exception as e:
                print(f"Warning: Error running trend_acceleration_plots_pooled.py: {e}")
            
            try:
                result = subprocess.run(['python', 'vreturns_analysis.py', permanent_folder], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"Warning: vreturns_analysis.py failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("Warning: vreturns_analysis.py timed out")
            except Exception as e:
                print(f"Warning: Error running vreturns_analysis.py: {e}")
                
            print(f"Iteration data saved to: {permanent_folder}")
            
            # Calculate fitness
            fitness_score = self.extract_metrics_and_calculate_fitness(trial_id, num_agents, permanent_folder)
            
            # Save iteration info
            iteration_info = {
                'trial_id': trial_id,
                'fitness_score': fitness_score,
                'parameters': {
                    'price_impact': price_impact,
                    'noise_std': noise_std,
                    'fundamental_proportion': fundamental_proportion,
                    'reversion_theta': reversion_theta,
                    'fundamental_sigma': fundamental_sigma,
                    'threshold_low': threshold_low,
                    'threshold_high': threshold_high,
                    'num_agents': num_agents
                },
                'simulation_settings': {
                    'num_steps': self.num_steps,
                    'num_repetitions': self.num_repetitions
                }
            }
            
            if hasattr(self, 'trial_metrics') and trial_id in self.trial_metrics:
                iteration_info['metrics'] = self.trial_metrics[trial_id]
            
            with open(os.path.join(permanent_folder, 'iteration_info.json'), 'w') as f:
                json.dump(iteration_info, f, indent=2, sort_keys=False)
            
            # Rename folder with fitness score
            try:
                new_folder_name = f"iteration_{trial_id}_fitness_{fitness_score:.4f}"
                new_folder_path = os.path.join(main_iterations_folder, new_folder_name)
                os.rename(permanent_folder, new_folder_path)
                permanent_folder = new_folder_path
                print(f"Renamed folder to: {new_folder_path}")
            except Exception as e:
                print(f"Error renaming folder: {e}")
            
            elapsed_time = time.time() - start_time
            print(f"Trial {trial_id} completed in {elapsed_time:.2f}s, fitness: {fitness_score:.4f}")
            
            return fitness_score
        
        except Exception as e:
            print(f"Error in trial {trial_id}: {e}")
            import traceback
            traceback.print_exc()
            fitness_score = -1000
            
            try:
                new_folder_name = f"iteration_{trial_id}_fitness_{fitness_score:.4f}"
                new_folder_path = os.path.join(main_iterations_folder, new_folder_name)
                if os.path.exists(permanent_folder):
                    os.rename(permanent_folder, new_folder_path)
                    permanent_folder = new_folder_path
            except Exception as rename_error:
                print(f"Error renaming folder: {rename_error}")
            
            return fitness_score
    
    def extract_metrics_and_calculate_fitness(self, trial_id, num_agents, experiments_folder):
        """Extract metrics and calculate fitness score.
        
        Args:
            trial_id (str): Trial identifier.
            num_agents (int): Number of agents used.
            experiments_folder (str): Path to experiment folder.
        
        Returns:
            float: Fitness score.
        """
        try:
            from scipy.stats import skew, kurtosis
            
            # Initialize metrics
            avg_kurtosis = 0
            avg_skew = 0
            geom_pvalue_up = 0
            geom_pvalue_down = 0
            vreturns_mean_pos = 0
            vreturns_mean_neg = 0
            
            if not os.path.exists(experiments_folder):
                print(f"Warning: Experiment folder not found: {experiments_folder}")
                return -1000
            
            exp_folder = experiments_folder
            
            # Load repetitions data
            repetitions_data = []
            rep_folders = []
            
            for item in os.listdir(exp_folder):
                if item.startswith('rep_') and os.path.isdir(os.path.join(exp_folder, item)):
                    rep_folders.append(item)
            
            rep_folders.sort(key=lambda x: int(x.split('_')[1]))
            
            for rep_folder in rep_folders:
                csv_path = os.path.join(exp_folder, rep_folder, 'price_series.csv')
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, index_col=0)
                        if len(df) > 500:
                            df_filtered = df.iloc[500:]
                            repetitions_data.append(df_filtered['market_price'])
                        else:
                            continue
                    except Exception as e:
                        print(f"Error loading {csv_path}: {e}")
                        continue
            
            if not repetitions_data:
                print("Warning: No repetition data found")
                return -1000
            
            # Calculate kurtosis and skewness
            pooled_returns = []
            for price_series in repetitions_data:
                log_returns = np.log(price_series[1:].values / price_series[:-1].values)
                pooled_returns.extend(log_returns)
            
            pooled_returns = np.array(pooled_returns)
            avg_kurtosis = kurtosis(pooled_returns)
            avg_skew = skew(pooled_returns)
            
            # Extract geometric p-values from trend analysis
            trend_plots_folder = os.path.join(exp_folder, 'trend_acceleration_plots')
            trend_file = None
            
            if os.path.exists(trend_plots_folder):
                for file in os.listdir(trend_plots_folder):
                    if file.endswith('_comprehensive_metrics.csv'):
                        trend_file = os.path.join(trend_plots_folder, file)
                        break
            
            if trend_file and os.path.exists(trend_file):
                try:
                    trend_df = pd.read_csv(trend_file)
                    
                    geom_up_rows = trend_df[
                        (trend_df['distribution'] == 'Geometric') & 
                        (trend_df['metric_name'] == 'Chi2_p_value') &
                        (trend_df['subplot'] == 'Ascending')
                    ]
                    geom_down_rows = trend_df[
                        (trend_df['distribution'] == 'Geometric') & 
                        (trend_df['metric_name'] == 'Chi2_p_value') &
                        (trend_df['subplot'] == 'Descending')
                    ]
                    
                    if not geom_up_rows.empty:
                        geom_pvalue_up = geom_up_rows['value'].iloc[0]
                    if not geom_down_rows.empty:
                        geom_pvalue_down = geom_down_rows['value'].iloc[0]
                        
                except Exception as e:
                    print(f"Error reading trend analysis: {e}")
            
            # Calculate VReturns and read gamma fit metrics from JSON
            vreturns_rmse_pos = 0
            vreturns_rmse_neg = 0
            vreturns_ks_pos = 0
            vreturns_ks_neg = 0
            
            try:
                from vreturns_analysis import extract_vreturns_from_repetitions
                
                all_vreturns = extract_vreturns_from_repetitions(repetitions_data)
                
                if len(all_vreturns) > 0:
                    positive_vreturns = all_vreturns[all_vreturns > 0]
                    negative_vreturns = all_vreturns[all_vreturns < 0]
                    
                    if len(positive_vreturns) > 0:
                        vreturns_mean_pos = np.mean(positive_vreturns)
                    
                    if len(negative_vreturns) > 0:
                        negative_abs = np.abs(negative_vreturns)
                        vreturns_mean_neg = np.mean(negative_abs)
                
                # Read optimal cuts and metrics from JSON file generated by vreturns_analysis.py
                cuts_json_path = os.path.join(exp_folder, 'vreturns_optimal_cuts_pooled.json')
                if os.path.exists(cuts_json_path):
                    try:
                        with open(cuts_json_path, 'r') as f:
                            cuts_data = json.load(f)
                            vreturns_rmse_pos = cuts_data.get('positive', {}).get('rmse', 0)
                            vreturns_rmse_neg = cuts_data.get('negative', {}).get('rmse', 0)
                            vreturns_ks_pos = cuts_data.get('positive', {}).get('ks_stat', 0)
                            vreturns_ks_neg = cuts_data.get('negative', {}).get('ks_stat', 0)
                    except Exception as e:
                        print(f"Warning: Could not read optimal cuts JSON: {e}")
                    
            except Exception as e:
                print(f"Error calculating VReturns: {e}")
            
            # Calculate fitness score
            avg_geom_pvalue = (geom_pvalue_up + geom_pvalue_down) / 2.0
            avg_vreturns_mean = (vreturns_mean_pos + vreturns_mean_neg) / 2.0
            avg_vreturns_rmse = (vreturns_rmse_pos + vreturns_rmse_neg) / 2.0
            avg_vreturns_ks = (vreturns_ks_pos + vreturns_ks_neg) / 2.0
            
            kurtosis_penalty = abs(avg_kurtosis - self.target_kurtosis) / 5.0
            skew_penalty = abs(avg_skew - self.target_skew) * 3.0
            vreturns_penalty = abs(avg_vreturns_mean - self.target_vreturns_mean) * 50.0
            vreturns_rmse_penalty = avg_vreturns_rmse * 50.0
            
            kurtosis_score = -kurtosis_penalty * 2.5
            skew_score = -skew_penalty
            geom_score = avg_geom_pvalue * 25
            vreturns_score = -vreturns_penalty
            vreturns_rmse_score = -vreturns_rmse_penalty
            
            geom_penalty = -50 if avg_geom_pvalue < self.min_geom_pvalue else 0
            minimum_kurtosis_penalty = -100 if avg_kurtosis < self.minimum_kurtosis else 0
            
            fitness_score = kurtosis_score + skew_score + geom_score + geom_penalty + vreturns_score + vreturns_rmse_score + minimum_kurtosis_penalty
            
            # Store metrics
            metrics = {
                'num_agents': num_agents,
                'kurtosis': avg_kurtosis,
                'skewness': avg_skew,
                'geom_pvalue_up': geom_pvalue_up,
                'geom_pvalue_down': geom_pvalue_down,
                'geom_pvalue_avg': avg_geom_pvalue,
                'vreturns_mean_pos': vreturns_mean_pos,
                'vreturns_mean_neg': vreturns_mean_neg,
                'vreturns_mean_avg': avg_vreturns_mean,
                'vreturns_rmse_pos': vreturns_rmse_pos,
                'vreturns_rmse_neg': vreturns_rmse_neg,
                'vreturns_rmse_avg': avg_vreturns_rmse,
                'vreturns_ks_pos': vreturns_ks_pos,
                'vreturns_ks_neg': vreturns_ks_neg,
                'vreturns_ks_avg': avg_vreturns_ks,
                'kurtosis_score': kurtosis_score,
                'skew_score': skew_score,
                'geom_score': geom_score,
                'geom_penalty': geom_penalty,
                'vreturns_score': vreturns_score,
                'vreturns_rmse_score': vreturns_rmse_score,
                'minimum_kurtosis_penalty': minimum_kurtosis_penalty,
                'fitness_score': fitness_score
            }
            
            if not hasattr(self, 'trial_metrics'):
                self.trial_metrics = {}
            self.trial_metrics[trial_id] = metrics
            
            return fitness_score
            
        except Exception as e:
            print(f"Error extracting metrics: {e}")
            
            if not hasattr(self, 'trial_metrics'):
                self.trial_metrics = {}
            self.trial_metrics[trial_id] = {
                'num_agents': num_agents,
                'kurtosis': 0,
                'skewness': 0,
                'geom_pvalue_up': 0,
                'geom_pvalue_down': 0,
                'geom_pvalue_avg': 0,
                'vreturns_mean_pos': 0,
                'vreturns_mean_neg': 0,
                'vreturns_mean_avg': 0,
                'vreturns_rmse_pos': 0,
                'vreturns_rmse_neg': 0,
                'vreturns_rmse_avg': 0,
                'vreturns_ks_pos': 0,
                'vreturns_ks_neg': 0,
                'vreturns_ks_avg': 0,
                'kurtosis_score': -1000,
                'skew_score': 0,
                'geom_score': 0,
                'geom_penalty': 0,
                'vreturns_score': -1000,
                'vreturns_rmse_score': -1000,
                'minimum_kurtosis_penalty': -50,
                'fitness_score': -1000,
                'error': str(e)
            }
            
            return -1000
    
    def _save_progress(self, optimizer, current_iteration, total_iterations):
        """Save optimization progress to JSON file."""
        try:
            best_fitness = optimizer.max['target']
            best_params = optimizer.max['params']
            
            results = {
                'best_fitness': float(best_fitness),
                'best_params': best_params,
                'current_iteration': current_iteration,
                'total_iterations_planned': total_iterations,
                'all_results': []
            }
            
            if hasattr(self, 'trial_metrics'):
                for trial_id, metrics in self.trial_metrics.items():
                    if abs(metrics.get('fitness_score', 0) - best_fitness) < 1e-6:
                        results['best_metrics'] = {
                            'num_agents': metrics['num_agents'],
                            'kurtosis': metrics['kurtosis'],
                            'skewness': metrics['skewness'],
                            'geom_pvalue_up': metrics['geom_pvalue_up'],
                            'geom_pvalue_down': metrics['geom_pvalue_down'],
                            'geom_pvalue_avg': metrics['geom_pvalue_avg'],
                            'vreturns_mean_pos': metrics['vreturns_mean_pos'],
                            'vreturns_mean_neg': metrics['vreturns_mean_neg'],
                            'vreturns_mean_avg': metrics['vreturns_mean_avg'],
                            'vreturns_rmse_pos': metrics.get('vreturns_rmse_pos', 0),
                            'vreturns_rmse_neg': metrics.get('vreturns_rmse_neg', 0),
                            'vreturns_rmse_avg': metrics.get('vreturns_rmse_avg', 0),
                            'vreturns_ks_pos': metrics.get('vreturns_ks_pos', 0),
                            'vreturns_ks_neg': metrics.get('vreturns_ks_neg', 0),
                            'vreturns_ks_avg': metrics.get('vreturns_ks_avg', 0)
                        }
                        break
            
            for i, (target, params_array) in enumerate(zip(optimizer.space.target, optimizer.space.params)):
                trial_result = {
                    'iteration': i + 1,
                    'fitness': float(target),
                    'params': dict(zip(optimizer.space.keys, params_array))
                }
                
                trial_id = f"trial_{i+1}"
                if hasattr(self, 'trial_metrics') and trial_id in self.trial_metrics:
                    metrics = self.trial_metrics[trial_id]
                    trial_result['metrics'] = {
                        'num_agents': metrics['num_agents'],
                        'kurtosis': metrics['kurtosis'],
                        'skewness': metrics['skewness'],
                        'geom_pvalue_up': metrics['geom_pvalue_up'],
                        'geom_pvalue_down': metrics['geom_pvalue_down'],
                        'geom_pvalue_avg': metrics['geom_pvalue_avg'],
                        'vreturns_mean_pos': metrics['vreturns_mean_pos'],
                        'vreturns_mean_neg': metrics['vreturns_mean_neg'],
                        'vreturns_mean_avg': metrics['vreturns_mean_avg'],
                        'vreturns_rmse_pos': metrics.get('vreturns_rmse_pos', 0),
                        'vreturns_rmse_neg': metrics.get('vreturns_rmse_neg', 0),
                        'vreturns_rmse_avg': metrics.get('vreturns_rmse_avg', 0),
                        'vreturns_ks_pos': metrics.get('vreturns_ks_pos', 0),
                        'vreturns_ks_neg': metrics.get('vreturns_ks_neg', 0),
                        'vreturns_ks_avg': metrics.get('vreturns_ks_avg', 0)
                    }
                
                results['all_results'].append(trial_result)
            
            with open('optimization_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Progress saved after iteration {current_iteration}/{total_iterations}")
            
        except Exception as e:
            print(f"Error saving progress: {e}")

    def optimize(self, init_points=100, n_iter=100):
        """Run Bayesian optimization.
        
        Args:
            init_points (int): Number of initial random points.
            n_iter (int): Number of optimization iterations.
        """
        # Define parameter bounds
        pbounds = {
            'price_impact': (0.0002, 0.0012),
            'noise_std': (0.005, 0.060),
            'fundamental_proportion': (0.3, 0.99),
            'reversion_theta': (0.01, 0.12),
            'fundamental_sigma': (0.001, 0.035),
            'threshold_low': (0.01, 0.10),
            'threshold_high': (0.04, 0.30),
            'num_agents': (200, 1000)
        }
        
        print("="*60)
        print("STARTING BAYESIAN OPTIMIZATION FOR FINANCIAL ABM")
        print("="*60)
        print(f"Target kurtosis: {self.target_kurtosis}")
        print(f"Target skewness: {self.target_skew}")
        print(f"Minimum geometric p-value: {self.min_geom_pvalue}")
        print(f"Parameter bounds:")
        for param, bounds in pbounds.items():
            print(f"  {param}: {bounds}")
        print(f"Optimization: {init_points} initial points + {n_iter} iterations")
        print("="*60)
        
        # Initialize optimizer
        optimizer = BayesianOptimization(
            f=self.run_simulation_and_analyze,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )
        
        # Run optimization
        start_time = time.time()
        iteration_counter = 0
        
        print(f"Running {init_points} initial random points...")
        for i in range(init_points):
            iteration_counter += 1
            optimizer.maximize(init_points=1, n_iter=0)
            self._save_progress(optimizer, iteration_counter, init_points + n_iter)
        
        print(f"Running {n_iter} intelligent iterations...")
        for i in range(n_iter):
            iteration_counter += 1
            optimizer.maximize(init_points=0, n_iter=1)
            self._save_progress(optimizer, iteration_counter, init_points + n_iter)
        
        total_time = time.time() - start_time
        
        # Print results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED!")
        print("="*60)
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Best fitness score: {optimizer.max['target']:.4f}")
        print("Best parameters:")
        for param, value in optimizer.max['params'].items():
            print(f"  {param}: {value:.6f}")
        
        # Collect results
        all_trials = []
        for i, (target, params) in enumerate(zip(optimizer.space.target, optimizer.space.params)):
            trial_result = {
                'iteration': i,
                'fitness': target,
                'params': dict(zip(optimizer.space.keys, params))
            }
            all_trials.append(trial_result)
        
        all_trials.sort(key=lambda x: x['fitness'], reverse=True)
        top_5_results = all_trials[:5]
        
        # Save results
        results = {
            'summary': {
                'total_iterations': len(all_trials),
                'best_fitness': optimizer.max['target'],
                'optimization_time_minutes': total_time / 60
            },
            'top_5_best_configurations': top_5_results,
            'all_results': all_trials
        }
        
        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to 'optimization_results.json'")
        print("="*60)
        
        return optimizer.max


def test_optimization():
    """Test optimization with minimal iterations."""
    print("="*60)
    print("RUNNING TEST OPTIMIZATION (MINIMAL ITERATIONS)")
    print("="*60)
    
    calibrator = ModelCalibrator(
        target_kurtosis=7.0,
        target_skew=0.0,
        min_geom_pvalue=0.05
    )
    
    best_params = calibrator.optimize(init_points=2, n_iter=1)
    
    print("Test completed successfully!")
    return best_params


def main():
    """Main execution function."""
    calibrator = ModelCalibrator(
        target_kurtosis=7.0,
        target_skew=0.0,
        min_geom_pvalue=0.05
    )
    
    best_params = calibrator.optimize(init_points=100, n_iter=100)
    
    return best_params


if __name__ == "__main__":
    main()
