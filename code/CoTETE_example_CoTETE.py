from contextlib import redirect_stdout
import time
import numpy as np
from juliacall import Pkg as jl_pkg
from juliacall import Main as jl
import torch
print("Julia runtime started via juliacall.")
project_path = "D:/Code/CoTETE.jl/"
print(f"Activating project at: {project_path}")
jl_pkg.activate(project_path)
print("Instantiating project dependencies...")
jl_pkg.instantiate()
print("Loading CoTETE module...")
jl.seval("using CoTETE")
CoTETE = jl.CoTETE
print("CoTETE module loaded successfully.")

from CoTETE_example_test import generate_spike_trains_CoTETE
import csv
seed=52
num_source_events = int(2e+4)
x_events, y_events, candidates, accepted = generate_spike_trains_CoTETE(RATE_Y = 1.0, RATE_X_MAX=6, NUM_Y_EVENTS=num_source_events,seed=seed)
print(f"Generated {len(x_events)} events for X and {len(y_events)} events for Y.")


results = []

for history_length in [256]:
    params = CoTETE.CoTETEParameters(l_x = history_length, l_y = history_length)
    for i in range(10):
        start_time = time.time()
        result = CoTETE.estimate_TE_from_event_times(params, jl.Array(np.array(y_events)), jl.Array(np.array(x_events)))
        end_time = time.time()
        duration = end_time - start_time
        print(f"history_length: {history_length}, Iteration {i+1}: TE = {result} nats/sec, Time = {duration:.4f}s")
        results.append({"history_length": history_length, "transfer_entropy": result, "runtime_seconds": duration})

        # Write results to CSV
        with open("results/cotete_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["history_length", "transfer_entropy", "runtime_seconds"])
            writer.writeheader()
            writer.writerows(results)

# total_duration = sum(r["runtime_seconds"] for r in results)

# print(f"\n--- CoTETE TE Estimation Completed in {duration/60:.2f} minutes ---")
# print(f"\n--- CoTETE Estimated Transfer Entropy: {result} nats per second ---")
