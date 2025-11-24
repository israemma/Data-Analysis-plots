import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

# Import only the market model function
from market_model import generate_market_data

def run_single_simulation(args):
    """
    Runs a single simulation (experiment + repetition combination).
    This function is designed to work with ProcessPoolExecutor.
    
    Args:
        args (tuple): (experiment_idx, repetition_idx, params, experiments_root_folder, num_agents, num_steps)
    """
    experiment_idx, repetition_idx, params, experiments_root_folder, num_agents, num_steps = args
    
    # Create experiment folder name
    name_parts = []
    for key, value in params.items():
        key_abbr = ''.join([c for c in key.title() if c.isupper()]) or key[:4]
        if isinstance(value, float):
            if key == 'price_impact':
                # Format price_impact with full precision
                formatted_value = f"{value:.10f}".rstrip('0').rstrip('.')
                if formatted_value.startswith('0.'):
                    formatted_value = formatted_value[2:]
                name_parts.append(f"{key_abbr}_{formatted_value}")
            else:
                name_parts.append(f"{key_abbr}_{value:.3f}")
        else:
            name_parts.append(f"{key_abbr}_{value}")
    
    descriptive_name = "_".join(name_parts)
    experiment_folder_name = f"exp_{experiment_idx}_{descriptive_name}".replace('.', '_')
    experiment_folder_path = os.path.join(experiments_root_folder, experiment_folder_name)
    
    # Create experiment folder if it doesn't exist
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    
    # Create repetition folder
    rep_folder_name = f"rep_{repetition_idx}"
    repetition_folder_path = os.path.join(experiment_folder_path, rep_folder_name)
    
    if not os.path.exists(repetition_folder_path):
        os.makedirs(repetition_folder_path)
    
    # Define the data file path for this repetition
    csv_file_path = os.path.join(repetition_folder_path, 'price_series.csv')
    
    # Generate market data
    generate_market_data(
        num_agents=num_agents,
        num_steps=num_steps,
        output_file=csv_file_path,
        market_params=params
    )
    
    return f"Exp {experiment_idx}, Rep {repetition_idx}"

def create_all_simulation_tasks(combinations, num_repetitions, experiments_root_folder, num_agents, num_steps):
    """
    Creates all simulation tasks (experiment + repetition combinations) for parallel execution.
    
    Args:
        combinations (list): List of parameter combinations for experiments
        num_repetitions (int): Number of repetitions per experiment
        experiments_root_folder (str): Root folder for all experiments
        num_agents (int): Number of agents
        num_steps (int): Number of simulation steps
    
    Returns:
        list: List of task tuples for ProcessPoolExecutor
    """
    all_tasks = []
    
    for exp_idx, params in enumerate(combinations):
        for rep_idx in range(1, num_repetitions + 1):
            task = (exp_idx + 1, rep_idx, params, experiments_root_folder, num_agents, num_steps)
            all_tasks.append(task)
    
    return all_tasks

def run_multi_experiments():
    """
    Runs multiple experiments with full parallelization using all CPU cores.
    All simulations (experiments + repetitions) run in parallel.
    """
    # General Simulation Parameters
    NUM_AGENTS = 1000
    NUM_STEPS = 2000
    NUM_REPETITIONS = 20
    EXPERIMENTS_ROOT_FOLDER = 'experiments_multi'
    
    # Get number of CPU cores and use all of them
    num_cores = mp.cpu_count()
    max_workers = num_cores
    
    # Create the root folder for all experiments if it doesn't exist
    if not os.path.exists(EXPERIMENTS_ROOT_FOLDER):
        os.makedirs(EXPERIMENTS_ROOT_FOLDER)
        print(f"Root folder created at: '{EXPERIMENTS_ROOT_FOLDER}'")
    
    # Define 100 parameter combinations for the uniform noise model (same as original)
    np.random.seed(42)  # For reproducibility
    combinations = []
    num_combinations = 100
    for _ in range(num_combinations):
        params = {
            'price_impact': np.random.uniform(low=0.0004, high=0.0009),
            'noise_std': np.random.uniform(low=0.005, high=0.05),
            'max_holding_period': np.random.randint(10, 100),
        }
        combinations.append(params)
    
    # Create all simulation tasks
    all_tasks = create_all_simulation_tasks(
        combinations, NUM_REPETITIONS, EXPERIMENTS_ROOT_FOLDER, NUM_AGENTS, NUM_STEPS
    )
    
    total_simulations = len(all_tasks)
    
    print(f"Testing {len(combinations)} parameter combinations with {NUM_REPETITIONS} repetitions each.")
    print(f"Total simulations: {total_simulations}")
    print(f"Using {max_workers} CPU cores")
    print("\n" + "="*60)
    
    overall_start_time = time.time()
    
    # Run ALL simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {}
        for task in all_tasks:
            future = executor.submit(run_single_simulation, task)
            future_to_task[future] = task
        
        # Monitor progress
        completed_simulations = 0
        failed_simulations = 0
        
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            exp_idx, rep_idx = task[0], task[1]
            
            try:
                result = future.result()
                completed_simulations += 1
                
                # Print progress every 100 completions
                if completed_simulations % 100 == 0:
                    progress_pct = (completed_simulations / total_simulations) * 100
                    elapsed = time.time() - overall_start_time
                    estimated_total = elapsed * total_simulations / completed_simulations
                    remaining = estimated_total - elapsed
                    
                    print(f"Progress: {completed_simulations}/{total_simulations} ({progress_pct:.1f}%) | "
                          f"Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s")
                
            except Exception as exc:
                failed_simulations += 1
                print(f"✗ Exp {exp_idx}, Rep {rep_idx} failed: {exc}")
    
    total_elapsed_time = time.time() - overall_start_time
    
    print("\n" + "="*60)
    print("SIMULATIONS COMPLETED")
    print("="*60)
    print(f"Total: {total_simulations} | Successful: {completed_simulations} | Failed: {failed_simulations}")
    print(f"Execution time: {total_elapsed_time/60:.2f} minutes ({total_elapsed_time/total_simulations:.2f}s per simulation)")

if __name__ == "__main__":
    run_multi_experiments()
