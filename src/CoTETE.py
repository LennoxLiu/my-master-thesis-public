import numpy as np
from juliacall import Pkg as jl_pkg
from juliacall import Main as jl

print("Julia runtime started via juliacall.")
project_path = "D:/Code/CoTETE.jl/"

# 1. Update the package registry
# This is the NEW, critical step to fix "no known versions"
print("Updating Julia package registry (this may take a moment)...")
jl_pkg.update()

# 2. Activate your project's environment
print(f"Activating project at: {project_path}")
jl_pkg.activate(project_path)

# 3. Instantiate (install) all dependencies
# This will read your Project.toml and install everything
print("Instantiating project dependencies...")
jl_pkg.instantiate()

# 4. Load your module
print("Loading CoTETE module...")
jl.seval("using CoTETE")
CoTETE = jl.CoTETE
print("CoTETE module loaded successfully.")

# # 5. Your original code
# params = CoTETE.CoTETEParameters(l_x = 1, l_y = 1)

# target = 1e3*np.random.rand(1000)
# target = np.sort(target)
# source = 1e3*np.random.rand(1000)
# source = np.sort(source)

# print("Running CoTETE.estimate_TE_from_event_times...")
# result = CoTETE.estimate_TE_from_event_times(params, jl.Array(target), jl.Array(source))
# print(f"Result: {result}")