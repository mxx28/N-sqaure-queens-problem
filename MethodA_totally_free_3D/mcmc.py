import numpy as np
from utility import random_state, compute_energy, compute_delta_energy, get_positions
from tqdm.auto import tqdm

class MCMC_result:
    """Container for storing the full MCMC trace and summary statistics."""
    def __init__(self, N, energies, betas, accepted_moves, final_state, min_energy, iterations, acceptance_curve):
        self.N = N
        self.energies = energies          # np.array of energies at each step
        self.betas = betas                # np.array of beta values
        self.accepted_moves = accepted_moves
        self.final_state = final_state    # final configuration
        self.min_energy = min_energy      # minimum energy observed
        self.iterations = iterations      # number of Metropolis updates performed
        self.acceptance_curve = acceptance_curve


# Metropolis update
def metropolis_step_single(state, occupied, N, beta, current_energy=None, rng=None):
    """
    Metropolis step for the totally-free representation.

    Args:
        state: array (num_queens, 3), will be copied as proposal
        N: board size (cube is N x N x N)
        beta: inverse temperature
        current_energy: if None, compute from scratch
        rng: numpy RNG

    Returns:
        proposal_state, new_energy, accepted
    """
    if rng is None:
        rng = np.random.default_rng()

    num_queens = state.shape[0]

    if current_energy is None:
        current_energy = compute_energy(state)

    # 1) Randomly select a queen to move
    q_idx = rng.integers(0, num_queens)
    old_pos = tuple(map(int, state[q_idx]))

    # 2) Randomly select an empty position
    
    # Sample an empty position
    while True:
        flat = int(rng.integers(N ** 3))
        i = flat // (N * N)
        rem = flat % (N * N)
        j = rem // N
        k = rem % N
        new_pos = (i, j, k)
        if new_pos not in occupied:
            break

    # 3) Compute ΔE
    delta = compute_delta_energy(state, q_idx, new_pos)
    new_energy = current_energy + delta

    # 4) Metropolis-Hastings acceptance
    if delta <= 0 or rng.random() < np.exp(-beta * delta):
        proposal = state.copy()
        proposal[q_idx] = new_pos

        occupied.remove(old_pos)
        occupied.add(new_pos)

        return proposal, occupied, new_energy, True

    return state, occupied, current_energy, False

        
def run_mcmc(
    N,
    beta=1.0,
    max_steps=20000,
    target_energy=0,
    seed=None,
    verbose=False,
    schedule="fixed",      # "fixed", "exponential", "geometric", "log"
    beta_end=None,
    cooling_rate=None,
    save_path=None,
):
    """
    Run a Metropolis MCMC / simulated annealing chain.

    Args:
        N: Board size (N x N base, heights in {0, ..., N-1}).
        beta: Inverse temperature parameter.
        max_steps: Maximum number of Metropolis updates.
        target_energy: Stop early if energy <= target_energy.
        seed: Random seed for reproducibility.
        schedule: 'fixed', 'exponential', 'geometric', or 'log'.
        beta_end: Final beta for 'geometric'/'log' schedules.
        cooling_rate: Factor (>1) for 'exponential' schedule.

    Returns:
        MCMC_result object containing the energy trace, final state, etc.
    """
    rng = np.random.default_rng(seed)

    state, occupied = random_state(N, rng)
    current_energy = compute_energy(state)

    energies = [current_energy]
    betas = []
    accepted = 0
    acceptance_curve = []

    for step in range(1, max_steps + 1):

        # Beta schedule
        if schedule == "fixed":
            beta_t = beta

        elif schedule == "exponential":
            if cooling_rate is None:
                raise ValueError("cooling_rate must be provided for schedule='exponential'")
            beta_t = beta * (cooling_rate ** step)
            # beta_t = beta * (cooling_rate ** (step / max_steps))

        elif schedule == "log":
            beta_t = beta * np.log(1 + step) / np.log(1 + max_steps)

        elif schedule == "geometric":
            beta_t = beta * (beta_end / beta) ** (step/max_steps)
            
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        betas.append(beta_t)

        # One Metropolis update (move one queen to an empty position)
        new_state, occupied, new_energy, accepted_flag = metropolis_step_single(
            state, occupied, N, beta_t, current_energy, rng
        )

        if accepted_flag:
            state = new_state
            current_energy = new_energy
            accepted += 1

        energies.append(current_energy)
        acceptance_curve.append(100.0 * accepted / step)

        if verbose and (step % 5000 == 0):
            print(
                f"step {step}: energy={current_energy}, beta={beta_t:.3f}, "
                f"acceptance={acceptance_curve[-1]:.1f}%"
            )

        if current_energy <= target_energy:
            if verbose:
                print(f"N={N}: Found energy {current_energy} at step {step}")
            break

    energies_arr = np.array(energies)
    betas_arr = np.array(betas)
    acc_arr = np.array(acceptance_curve)

    if save_path is not None:
        # Convert state to queen positions (x, y, z) and save as text file
        positions = get_positions(state)
        with open(save_path, 'w') as f:
            for x, y, z in positions:
                f.write(f"{x},{y},{z}\n")

    return MCMC_result(
        N=N,
        energies=energies_arr,
        betas=betas_arr,
        accepted_moves=accepted,
        final_state=state,
        min_energy=int(energies_arr.min()),
        iterations=len(betas_arr),
        acceptance_curve=acc_arr,
    )

def average_energy_over_runs(
    N,
    beta,
    beta_end=None,
    max_steps=20000,
    runs=3,
    schedule="fixed",
    base_seed=None,
    cooling_rate=1.001,
):
    """
    Run multiple MCMC chains and return:
      - per-run energy traces (padded to length max_steps+1)
      - mean energy per step
      - std of energy per step
      - number of runs that reached energy 0

    Args:
        N: Board size.
        beta: Inverse temperature (for annealing schedules, this is the initial beta).
        beta_end: Final beta for 'geometric' / 'log' schedules.
        max_steps: Maximum number of Metropolis updates per run.
        runs: Number of independent runs.
        schedule: 'fixed', 'exponential', 'geometric', or 'log'.
        base_seed: Base seed; if not None, seeds are base_seed + run_idx.
        cooling_rate: Growth factor for 'exponential' schedule.

    Returns:
        N: board size (for convenience)
        energies_matrix: array of shape (runs, max_steps+1)
        mean_energy:     array of shape (max_steps+1,)
        std_energy:      array of shape (max_steps+1,)
        num_zero_runs:   how many runs reached energy 0 at some point
    """
    traces = []
    num_zero_runs = 0

    it = range(runs)
    it = tqdm(it, desc=f"N={N}, schedule={schedule}", leave=False)

    for run_idx in it:
        seed = None if base_seed is None else base_seed + run_idx

        trace = run_mcmc(
            N=N,
            beta=beta,
            max_steps=max_steps,
            target_energy=0,   # stop as soon as energy reaches 0
            seed=seed,
            schedule=schedule,
            cooling_rate=cooling_rate,
            beta_end=beta_end,
            verbose=False,
        )

        energies = trace.energies  # length <= max_steps+1
        if trace.min_energy == 0:
            num_zero_runs += 1

        # pad to length max_steps+1 with the last value (if stopped early)
        if energies.size < max_steps + 1:
            pad_len = max_steps + 1 - energies.size
            last = energies[-1]
            energies = np.concatenate([energies, np.full(pad_len, last, dtype=energies.dtype)])

        traces.append(energies)

    energies_matrix = np.vstack(traces)
    mean_energy = energies_matrix.mean(axis=0)
    std_energy = energies_matrix.std(axis=0)
    final_energies = energies_matrix[:, -1]  # last energy of each run
    print(f"N={N}, num_zero_runs={num_zero_runs}, final_energies={final_energies.tolist()}")

    return N, energies_matrix, mean_energy, std_energy