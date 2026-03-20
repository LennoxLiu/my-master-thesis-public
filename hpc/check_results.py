import os
from pathlib import Path

def check_missing_results(base_dirs, task_range):
    """
    Checks for the existence of config files in specified directories.
    
    Args:
        base_dirs (dict): Dictionary mapping model types to their directory paths.
        task_range (range): Range of task IDs to check.
    """
    report = {}

    for model_type, directory in base_dirs.items():
        missing_ids = []
        dir_path = Path(directory)
        
        # Determine prefix based on directory name or model type
        prefix = "opt_full" if "full" in model_type.lower() else "opt_reduced"
        
        for task_id in task_range:
            # Construct expected filename
            filename = f"{prefix}_{task_id}_best_config.txt"
            file_path = dir_path / filename
            
            if not file_path.exists():
                missing_ids.append(task_id)
        
        report[model_type] = {
            "path": directory,
            "missing": missing_ids,
            "total_expected": len(task_range)
        }

    return report

def print_summary(report):
    print(f"{'Model Type':<15} | {'Missing':<8} | {'Status'}")
    print("-" * 40)
    
    all_complete = True
    for model, data in report.items():
        missing_count = len(data['missing'])
        status = "COMPLETE" if missing_count == 0 else f"MISSING {missing_count}"
        print(f"{model:<15} | {missing_count:<8} | {status}")
        
        if missing_count > 0:
            all_complete = False
            # Print missing IDs in chunks for readability
            ids_str = ", ".join(map(str, data['missing']))
            if missing_count > 20:
                print(f"  > First 20 missing IDs: {', '.join(map(str, data['missing'][:20]))}...")
            else:
                print(f"  > Missing IDs: {ids_str}")
    
    print("-" * 40)
    if all_complete:
        print("Success: All result files are accounted for.")
    else:
        print("Action Required: Some tasks need to be re-run.")

if __name__ == "__main__":
    # Define directories and range
    directories = {
        "Full Model": "results/opt-full",
        "Reduced Model": "results/opt-reduced"
    }
    tasks = range(408)  # 0 to 407

    results_report = check_missing_results(directories, tasks)
    print_summary(results_report)