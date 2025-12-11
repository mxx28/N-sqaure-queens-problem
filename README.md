# N²-Queens Problem with MCMC

This repository implements **Markov Chain Monte Carlo** (MCMC) methods for the 3D (N²)-Queens problem using the **Metropolis–Hastings** algorithm with **simulated annealing**.

We provide **three different state representations** (and corresponding move proposals) that explore the same energy landscape in different ways.



## Problem Description

The **3D (N²)-Queens problem** extends the classic N-Queens puzzle to three dimensions.

* **Board**: an `N × N × N` cube.

* **Pieces**: `N²` queens.

* **Objective**: **minimize** the number of attacking queen pairs (ideally reach 0).

Two queens attack each other if they lie on the same:

* **Axis line** (sharing 2 coordinates, differing in the 3rd),

* **2D diagonal** in a coordinate plane,

* **3D space diagonal**.


## Run Instructions

1. **Run a single MCMC chain for a given configuration:**

   ```python
   # First, navigate to the folder containing the state representation you want to use:
   # - MethodA_totally_free_3D/ (State Rep 1: Totally Free 3D Coordinates)
   # - MethodB_column_height/ (State Rep 2: Column Height Matrix)
   # - MethodC_row_wise_permutation/ (State Rep 3: Row-Wise Permutation Heights)
   
   from mcmc import run_mcmc
   from plot import plot_energy_curve
   
   # Run MCMC with geometric annealing schedule
   trace = run_mcmc(
       N=11,
       beta=1.0,
       beta_end=1.5,
       max_steps=200000,
       schedule="geometric",
       verbose=True,
       seed=123,
       # Optional: save final configuration
       save_path="solution_N11.csv"  
   )
   
   # Visualize the energy curve
   plot_energy_curve(trace)
   
   print(f"Minimum energy reached: {trace.min_energy}")
   ```

2. **Run multiple MCMC chains and average the results:**

   ```python
   from mcmc import average_energy_over_runs
   from plot import plot_energy_curve_average
   
   # Run multiple independent MCMC chains with the same configuration
   N, energies_matrix, mean_energy, std_energy = average_energy_over_runs(
       N=11,
       beta=1.0,
       beta_end=1.5,
       max_steps=200000,
       runs=10,  # Number of independent runs
       schedule="geometric",
       base_seed=123  # Base seed for reproducibility
   )
   
   # Visualize average energy curve with individual runs
   plot_energy_curve_average(N, energies_matrix, mean_energy, std_energy)
   
   # The function automatically prints:
   # - Number of runs that reached energy 0
   # - Final energies for each run
   ```

3. **Run analysis notebooks**

   Each method folder contains a Jupyter notebook (`main_*.ipynb`) with example analyses:

   ```python
   # Navigate to the desired method folder
   cd MethodA_totally_free_3D/  # or MethodB_column_height/ or MethodC_row_wise_permutation/
   
   # Open and run the notebook
   # The notebook includes:
   # - Example MCMC runs with different annealing schedules
   # - Energy curve visualizations
   # - Comparison of different hyperparameters
   # - Analysis of convergence behavior
   ```

   Each notebook demonstrates:
   * Single MCMC chain runs with visualization
   * Multiple runs with averaged results
   * Comparison of with and without annealing schedules
   * Analysis of N and minimal energy reached


## Three State Representations & Moves

All three methods share the **same energy function** and **acceptance rule**; they differ only in **how a state is represented** and **how proposals are generated**.


### 1. Method A — Totally Free 3D Coordinates (State Rep 1)

* **State representation**

  A NumPy array of shape `(N², 3)`.

  Each row is a queen coordinate `(i, j, k)`, with all positions distinct:

  ```python
  state[q] = (i_q, j_q, k_q)  # 0 ≤ i_q, j_q, k_q < N
  ```

* **Constraints**

  No extra structural constraints: any distinct placement of `N²` queens is allowed.

* **Move (proposal)**

  1. Pick a queen index `q` uniformly at random.

  2. Sample a new cell `(i', j', k')` uniformly among all **unoccupied** cells.

  3. Propose moving queen `q` from `(i, j, k)` to `(i', j', k')`.

* **Energy update**

  The energy is the number of attacking pairs. For a move of a **single** queen, we compute `ΔE` by comparing the conflicts of that queen **before** vs **after** the move.

This is the **most flexible** representation, exploring the full configuration space without any structural bias.

---

### 2. Method B — Column Height Matrix (State Rep 2)

* **State representation**

  An `N × N` matrix `H`, where each entry stores a height:

  ```python
  H[i, j] = k   # queen at (i, j, k)
  ```

  Here each base cell `(i, j)` has **exactly one queen** somewhere along the `k`-axis. That is:

  * Exactly `N²` queens,

  * One per vertical "column" `(i, j)`.

* **Move (proposal)**

  1. Pick a random base cell `(i, j)`.

  2. Propose changing its height from `k` to `k'`

     (e.g., uniformly from `{0, ..., N-1}`, optionally with extra constraints).

  3. The queen at `(i, j, k)` moves vertically to `(i, j, k')`.

* **Energy update**

  Only one queen's coordinate changes, so we compute `ΔE` by checking conflicts of that queen **before** and **after** the height change.

This representation **removes a lot of degrees of freedom** (you never leave a column empty, and you never put 2 queens in the same column), making the search more structured and typically reducing the number of trivial conflicts.

---

### 3. Method C — Row-Wise Permutation Heights (State Rep 3)

* **State representation**

  Again an `N × N` height matrix `H`, but now each row is a **permutation** of `{0, 1, ..., N-1}`:

  ```python
  # For each row i:
  H[i, :] is a permutation of [0, 1, ..., N-1]
  ```

  That is:

  * Exactly one queen in each base cell `(i, j)`,

  * For each fixed row `i`, all heights `k` appear exactly once along that row.

* **Move (proposal)**

  1. Pick a row index `i` uniformly at random.

  2. Pick two distinct columns `j1 ≠ j2` uniformly at random.

  3. **Swap** the heights in that row:

     ```python
     H[i, j1], H[i, j2] = H[i, j2], H[i, j1]
     ```

     This moves the two queens:

     * `(i, j1, k1) → (i, j1, k2)`

     * `(i, j2, k2) → (i, j2, k1)`

* **Energy update**

  Only **two queens** move; `ΔE` is computed by checking conflicts involving these two queens **before** and **after** the swap.

Because each row is a permutation, the configuration is already relatively structured: rows are "spread" across heights. This often yields **lower initial energy** and moves that tweak conflicts more locally, which can help simulated annealing converge faster.


## Metropolis–Hastings & Simulated Annealing

We target the Gibbs/Boltzmann distribution:

$$
\pi(s) = \frac{1}{Z} e^{-\beta E(s)}
$$

where:

* `E(s)` is the energy (number of attacking pairs),

* `β` is the inverse temperature,

* `Z` is the partition function.

At each iteration:

1. **Propose** a new state `s'` from the current state `s` using one of the three move types above.

2. **Compute** the energy difference `ΔE = E(s') - E(s)`.

3. **Accept** with probability

$$
\alpha(s \to s') = \min\left(1, e^{-\beta \Delta E}\right)
$$

This is the standard Metropolis–Hastings rule with a **symmetric** proposal distribution.

### Annealing Schedules

The inverse temperature `β` is updated over time according to different schedules:

* **Fixed**:

$$
\beta(t) = \beta_0
$$

* **Exponential**:

$$
\beta(t) = \beta_0 \cdot c^t, \quad c > 1
$$

* **Geometric**:

$$
\beta(t) = \beta_0 \cdot \left(\frac{\beta_{\text{end}}}{\beta_0}\right)^{t/T_{\max}}
$$

where `β(t)` increases smoothly from `β₀` to `β_end` over `T_max` iterations.

* **Logarithmic**:

$$
\beta(t) = \beta_0 \cdot \frac{\log(1 + t)}{\log(1 + T_{\max})}
$$

Early iterations (small `β`) allow broad exploration; later iterations (large `β`) concentrate around low-energy configurations.
 