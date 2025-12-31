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
seed=42
num_source_events = int(1e+5)
x_events, y_events, candidates, accepted = generate_spike_trains_CoTETE(RATE_Y = 1.0, RATE_X_MAX=6, NUM_Y_EVENTS=num_source_events,seed=seed)
print(f"Generated {len(x_events)} events for X and {len(y_events)} events for Y.")
start_time = time.time()
params = CoTETE.CoTETEParameters(l_x = 4, l_y = 4)
result = CoTETE.estimate_TE_from_event_times(params, jl.Array(np.array(y_events)), jl.Array(np.array(x_events)))
end_time = time.time()
duration = end_time - start_time
print(f"\n--- CoTETE TE Estimation Completed in {duration/60:.2f} minutes ---")
print(f"\n--- CoTETE Estimated Transfer Entropy: {result} nats per second ---")
