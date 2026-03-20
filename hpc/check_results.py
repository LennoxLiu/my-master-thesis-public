import argparse
import pandas as pd
from pathlib import Path

def get_completed_runs(file_path):
    """Counts rows in the CSV to verify completed runs."""
    if not file_path.exists():
        return 0
    try:
        # Using low_memory=False to handle potentially large files silently
        df = pd.read_csv(file_path)
        return len(df) if not df.empty else 0
    except Exception:
        return 0

def scan_tasks(config, target_runs):
    """Scans for opt files and optionally run files."""
    results = {}

    for model_name, settings in config.items():
        opt_missing = []
        run_missing = []
        run_incomplete = []
        
        task_range = settings["range"]
        opt_dir = Path(settings["opt_path"])
        run_dir = Path(settings["runs_path"])

        for task_id in task_range:
            # Always check Optimization File (.txt)
            opt_file = opt_dir / f"{settings['opt_prefix']}_{task_id}_best_config.txt"
            if not opt_file.exists():
                opt_missing.append(task_id)

            # Only check Runs File (.csv) if target_runs is specified
            if target_runs is not None:
                run_file = run_dir / f"{settings['runs_prefix']}_{task_id}.csv"
                completed = get_completed_runs(run_file)
                
                if completed == 0:
                    run_missing.append(task_id)
                elif completed < target_runs:
                    run_incomplete.append((task_id, completed))

        results[model_name] = {
            "total": len(task_range),
            "opt_missing": opt_missing,
            "run_missing": run_missing if target_runs else None,
            "run_incomplete": run_incomplete if target_runs else None
        }
    return results

def print_report(results, target_runs):
    for model, data in results.items():
        print(f"\n{'='*20} {model.upper()} {'='*20}")
        print(f"Total Tasks Expected: {data['total']}")

        # --- Optimization Section ---
        print(f"\n[1] OPTIMIZATION (Best Configs)")
        if not data["opt_missing"]:
            print("  Status: COMPLETE")
        else:
            print(f"  Status: {len(data['opt_missing'])} MISSING")
            print(f"  IDs: {data['opt_missing'][:15]}{'...' if len(data['opt_missing']) > 15 else ''}")

        # --- Multiple Runs Section (Conditional) ---
        if target_runs is not None:
            print(f"\n[2] MULTIPLE RUNS (Target: {target_runs})")
            
            if not data["run_missing"] and not data["run_incomplete"]:
                print("  Status: COMPLETE")
            else:
                if data["run_missing"]:
                    print(f"  Missing Entirely: {len(data['run_missing'])} tasks")
                    print(f"    IDs: {data['run_missing'][:15]}{'...' if len(data['run_missing']) > 15 else ''}")
                
                if data["run_incomplete"]:
                    print(f"  Incomplete: {len(data['run_incomplete'])} tasks")
                    details = [f"{tid}({c}/{target_runs})" for tid, c in data['run_incomplete'][:8]]
                    print(f"    IDs: {', '.join(details)}{'...' if len(data['run_incomplete']) > 8 else ''}")
        else:
            print("\n[2] MULTIPLE RUNS: Skipped (use --num_runs to check)")

    print("\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Opt results, and optionally Run results.")
    parser.add_argument("--full", type=int, help="Task count for Full model")
    parser.add_argument("--reduced", type=int, help="Task count for Reduced model")
    # Set default to None to make it an explicit toggle
    parser.add_argument("--num_runs", type=int, default=None, help="Required repetitions in CSV")
    
    args = parser.parse_args()

    job_config = {}
    if args.full is not None:
        job_config["Full Model"] = {
            "range": range(args.full),
            "opt_path": "results/opt-full",
            "opt_prefix": "opt_full",
            "runs_path": "results/runs-full",
            "runs_prefix": "runs_full"
        }
    if args.reduced is not None:
        job_config["Reduced Model"] = {
            "range": range(args.reduced),
            "opt_path": "results/opt-reduced",
            "opt_prefix": "opt_reduced",
            "runs_path": "results/runs-reduced",
            "runs_prefix": "runs_reduced"
        }

    if not job_config:
        print("Error: No tasks specified. Use --full [N] or --reduced [N].")
    else:
        results = scan_tasks(job_config, args.num_runs)
        print_report(results, args.num_runs)