import argparse
from pathlib import Path

def check_missing_results(config):
    """
    Checks for missing files based on model-specific ranges.
    """
    report = {}

    for model_name, settings in config.items():
        missing_ids = []
        dir_path = Path(settings["path"])
        task_range = settings["range"]
        prefix = settings["prefix"]
        
        if not dir_path.exists():
            report[model_name] = {"missing": list(task_range), "status": "DIR_NOT_FOUND"}
            continue

        for task_id in task_range:
            filename = f"{prefix}_{task_id}_best_config.txt"
            if not (dir_path / filename).exists():
                missing_ids.append(task_id)
        
        report[model_name] = {
            "missing": missing_ids,
            "total_expected": len(task_range),
            "status": "OK"
        }

    return report

def print_summary(report):
    if not report:
        print("No models selected for scanning. Use --full or --reduced to specify ranges.")
        return

    print(f"\n{'Model Type':<15} | {'Missing':<8} | {'Total'} | {'Status'}")
    print("-" * 55)
    
    for model, data in report.items():
        missing_count = len(data['missing'])
        total = data['total_expected']
        
        if data.get("status") == "DIR_NOT_FOUND":
            status_msg = "DIR MISSING"
        else:
            status_msg = "COMPLETE" if missing_count == 0 else f"{missing_count} MISSING"
            
        print(f"{model:<15} | {missing_count:<8} | {total:<5} | {status_msg}")
        
        if 0 < missing_count <= 20:
            print(f"  > IDs: {', '.join(map(str, data['missing']))}")
        elif missing_count > 20:
            print(f"  > First 20 IDs: {', '.join(map(str, data['missing'][:20]))}...")
    
    print("-" * 55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan result folders for missing task output.")
    
    # Arguments set to None by default to make them optional
    parser.add_argument("--full", type=int, help="Task count for Full model (e.g., 408)")
    parser.add_argument("--reduced", type=int, help="Task count for Reduced model (e.g., 200)")
    
    parser.add_argument("--full_path", type=str, default="results/opt-full")
    parser.add_argument("--reduced_path", type=str, default="results/opt-reduced")
    
    args = parser.parse_args()

    # Build config only for provided arguments
    job_config = {}
    
    if args.full is not None:
        job_config["Full Model"] = {
            "path": args.full_path,
            "prefix": "opt_full",
            "range": range(args.full)
        }
        
    if args.reduced is not None:
        job_config["Reduced Model"] = {
            "path": args.reduced_path,
            "prefix": "opt_reduced",
            "range": range(args.reduced)
        }

    results_report = check_missing_results(job_config)
    print_summary(results_report)