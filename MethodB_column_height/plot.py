import numpy as np
import matplotlib.pyplot as plt
from mcmc import average_energy_over_runs, run_mcmc
from tqdm.auto import tqdm

def plot_energy_curve(result):
    steps = np.arange(len(result.energies))

    fig, axes = plt.subplots(3, 1, figsize=(5, 5), sharex=True)

    # Energy trace
    axes[0].plot(steps, result.energies, color='steelblue', linewidth=1.5, label='energy')
    axes[0].set_ylabel('Energy')
    axes[0].set_title(f'Energy/Beta/Acceptance (N={result.N})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Beta curve
    axes[1].plot(result.betas, color='coral', linewidth=1.5, label='beta')
    axes[1].set_ylabel('Beta')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Acceptance rate curve (stored directly in result)
    acc_curve = result.acceptance_curve
    step_axis = np.arange(acc_curve.size)
    axes[2].plot(step_axis, acc_curve, color='seagreen', linewidth=1.2, label='acc%')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Acc%')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def plot_energy_curve_average(N, energies_matrix, mean_energy, std_energy):
    plt.figure(figsize=(6, 3))
    
    # Plot each run's energy curve (with low alpha for visibility)
    num_runs = energies_matrix.shape[0]
    for i in range(num_runs):
        plt.plot(energies_matrix[i], alpha=0.3, linewidth=0.8, color='gray', label='individual runs' if i == 0 else '')
    
    # Plot mean energy (more prominent)
    plt.plot(mean_energy, label='mean energy', linewidth=2, color='steelblue')
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title(f'Average energy trace over {num_runs} runs (N={N})')
    plt.grid(True, alpha=0.3)
    plt.legend()

def compare_annealing_effect(N, beta_fix, beta_annealing_start, beta_annealing_end, max_steps, runs, base_seed=None, cooling_rate=1.001):
    setups = [
        ("fixed β={}".format(beta_fix), "fixed", beta_fix),
        # ("fixed β={}".format(beta_fix2), "fixed", beta_fix2),
        ("geometric start from β={} to β={}".format(beta_annealing_start, beta_annealing_end), "geometric", beta_annealing_start, beta_annealing_end),
    ]

    plt.figure(figsize=(7, 4))
    steps = np.arange(max_steps + 1)
    for setup in setups:
        label = setup[0]
        schedule = setup[1]
        if len(setup) == 3:
            # Fixed schedule: (label, "fixed", beta)
            beta = setup[2]
            beta_end = None
        else:
            # Geometric schedule: (label, "geometric", beta_start, beta_end)
            beta = setup[2]
            beta_end = setup[3]
        
        _, energies_matrix, mean_energy, std_energy = average_energy_over_runs(
            N,
            beta,
            beta_end=beta_end,
            max_steps=max_steps,
            runs=runs,
            schedule=schedule,
            base_seed=base_seed,
            cooling_rate=cooling_rate,
        )
        plt.plot(mean_energy, label=label)
        plt.fill_between(
            steps,
            np.clip(mean_energy - std_energy, a_min=0, a_max=None),
            mean_energy + std_energy,
            alpha=0.15,
        )

    plt.xlabel("Step")
    plt.ylabel("Mean energy ± std")
    plt.title(f"Annealing comparison over {runs} runs (N={N})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def min_energy_vs_N(
    N_values,
    beta=1.0,
    beta_end=None,
    max_steps=20000,
    runs=5,
    schedule="geometric",
    cooling_rate=1.001,
    target_energy=0,
    base_seed=None,
):
    """Run multiple MCMC chains per N with annealing and record min_energy for each run."""
    Ns = list(N_values)
    all_min_energies = []  # List of lists: each inner list contains min_energies for all runs of one N

    for idx, N in enumerate(tqdm(Ns, desc="Processing N values")):
        min_energies_for_N = []
        for run_idx in range(runs):
            seed = None if base_seed is None else base_seed + idx * runs + run_idx
            trace = run_mcmc(
                N=N,
                beta=beta,
                beta_end=beta_end,
                max_steps=max_steps,
                target_energy=target_energy,
                seed=seed,
                schedule=schedule,
                cooling_rate=cooling_rate,
                verbose=False,
            )
            min_energies_for_N.append(trace.min_energy)
        all_min_energies.append(min_energies_for_N)

    return {
        "N_values": np.array(Ns),
        "all_min_energies": all_min_energies,  # List of lists
        "runs": runs,
        "target_energy": target_energy,
        "schedule": schedule,
    }


def plot_min_energy_vs_N(result):
    Ns = result["N_values"]
    all_min_energies = result["all_min_energies"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot each run's min_energy for each N
    for idx, N in enumerate(Ns):
        min_energies = all_min_energies[idx]
        # Plot individual points with some transparency
        ax.scatter([N] * len(min_energies), min_energies, 
                  alpha=0.4, s=30, color='gray', zorder=1)
        # Plot mean
        mean_min = np.mean(min_energies)
        ax.scatter(N, mean_min, marker='o', s=100, color='steelblue', zorder=2, 
                  label='mean' if idx == 0 else '')
    
    # Plot mean line
    mean_mins = [np.mean(mins) for mins in all_min_energies]
    ax.plot(Ns, mean_mins, marker='o', color='steelblue', linewidth=2, 
           markersize=8, label='mean min energy', zorder=3)
    
    ax.set_xlabel("Board size N")
    ax.set_ylabel("Min energy")
    ax.set_title(f"Min energy vs N (annealing, {result['runs']} runs per N)")
    ax.grid(True, alpha=0.3)
    
    if result["target_energy"] == 0:
        ax.axhline(0, color="green", linestyle="--", linewidth=1, label="target (0)")
    
    ax.legend()
    plt.tight_layout()
    plt.show()