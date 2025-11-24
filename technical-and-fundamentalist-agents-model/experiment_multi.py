import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

from market_model import generate_market_data


def run_single_repetition(experiment_idx, repetition_idx, params, base_folder, num_agents, num_steps):
    """Execute a single experiment repetition.
    
    Args:
        experiment_idx (int): Experiment index (1-based).
        repetition_idx (int): Repetition index (1-based).
        params (dict): Market parameters for this experiment.
        base_folder (str): Base folder for experiment output.
        num_agents (int): Number of agents in the market.
        num_steps (int): Number of simulation steps.
    
    Returns:
        str: Completion status message.
    """
    # Create repetition folder
    rep_folder_name = f"rep_{repetition_idx}"
    repetition_folder_path = os.path.join(base_folder, rep_folder_name)
    
    if not os.path.exists(repetition_folder_path):
        os.makedirs(repetition_folder_path)
    
    # Generate and save market data
    csv_file_path = os.path.join(repetition_folder_path, 'price_series.csv')
    generate_market_data(
        num_agents=num_agents,
        num_steps=num_steps,
        output_file=csv_file_path,
        market_params=params
    )
    
    return f"Exp {experiment_idx}, Rep {repetition_idx}"


def run_experiment_repetitions(experiment_idx, params, experiments_root_folder, num_agents, num_steps, num_repetitions=20, random_seed=None):
    """Execute all repetitions for a single experiment.
    
    Args:
        experiment_idx (int): Experiment index (1-based).
        params (dict): Market parameters for this experiment.
        experiments_root_folder (str): Root folder for all experiments.
        num_agents (int): Number of agents in the market.
        num_steps (int): Number of simulation steps.
        num_repetitions (int): Number of repetitions per experiment.
        random_seed (int): Random seed for reproducibility.
    """
    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed(88)
    
    # Display experiment parameters
    print("\n" + "="*50)
    print(f"STARTING EXPERIMENT {experiment_idx} WITH {num_repetitions} REPETITIONS")
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        elif isinstance(value, tuple):
            print(f"  {key}: ({value[0]:.4f}, {value[1]:.4f})")
        else:
            print(f"  {key}: {value}")
    print("="*50)
    
    # Create experiment folder
    experiment_folder_path = experiments_root_folder
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    
    # Run repetitions in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=20) as executor:
        # Submit all repetition tasks
        future_to_rep = {}
        for rep_idx in range(1, num_repetitions + 1):
            future = executor.submit(
                run_single_repetition, 
                experiment_idx, 
                rep_idx, 
                params, 
                experiment_folder_path, 
                num_agents, 
                num_steps
            )
            future_to_rep[future] = rep_idx
        
        # Wait for completion
        for future in as_completed(future_to_rep):
            try:
                result = future.result()
            except Exception as exc:
                pass
    
    # Report completion
    elapsed_time = time.time() - start_time
    print(f"--- Experiment {experiment_idx} finished in {elapsed_time:.2f}s. Results in: '{experiment_folder_path}' ---")


def run_multi_experiments():
    """Execute multiple experiments with parameter combinations."""
    # Simulation parameters
    NUM_AGENTS = 800
    NUM_STEPS = 4000
    NUM_REPETITIONS = 20
    EXPERIMENTS_ROOT_FOLDER = 'experiments_multi'
    
    # Create root folder
    if not os.path.exists(EXPERIMENTS_ROOT_FOLDER):
        os.makedirs(EXPERIMENTS_ROOT_FOLDER)
    
    # Generate parameter combinations
    np.random.seed(88)
    combinations = []
    num_combinations = 100
    for _ in range(num_combinations):
        params = {
            'price_impact': np.random.uniform(low=0.00045, high=0.0005),
            'noise_std': np.random.uniform(low=0.020, high=0.025),
            'fundamental_proportion': np.random.uniform(low=0.85, high=0.90),
            'reversion_theta': np.random.uniform(low=0.04, high=0.055),
            'fundamental_sigma': np.random.uniform(low=0.005, high=0.015),
            'fundamental_threshold_range': (np.random.uniform(0.03, 0.05), np.random.uniform(0.09, 0.12))
        }
        combinations.append(params)
    
    # Execute experiments
    overall_start_time = time.time()
    
    for i, params in enumerate(combinations):
        run_experiment_repetitions(
            experiment_idx=i+1,
            params=params,
            experiments_root_folder=EXPERIMENTS_ROOT_FOLDER,
            num_agents=NUM_AGENTS,
            num_steps=NUM_STEPS,
            num_repetitions=NUM_REPETITIONS
        )
    
    # Report summary
    total_elapsed_time = time.time() - overall_start_time
    print(f"\n🎉 ALL {len(combinations)} EXPERIMENTS COMPLETED!")
    print(f"Total execution time: {total_elapsed_time:.2f}s ({total_elapsed_time/60:.2f} minutes)")
    print(f"Total simulations run: {len(combinations) * NUM_REPETITIONS}")


if __name__ == "__main__":
    run_multi_experiments()
