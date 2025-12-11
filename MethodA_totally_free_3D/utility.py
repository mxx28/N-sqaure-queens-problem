import numpy as np

########################################################
# Utility functions for the N^2 Queens problem
# Totally free representation:
#   state: array of shape (num_queens, 3), each row = (i, j, k)
########################################################


def random_state(N, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    num_queens = N * N
    total_cells = N ** 3

    flat_indices = rng.choice(total_cells, size=num_queens, replace=False)

    state = np.empty((num_queens, 3), dtype=int)
    for idx, flat in enumerate(flat_indices):
        i = flat // (N * N)
        rem = flat % (N * N)
        j = rem // N
        k = rem % N
        state[idx] = (int(i), int(j), int(k))

    # occupied set, built only once
    occupied = set(tuple(p) for p in state)
    return state, occupied


def get_positions(state):
    """
    Return queen coordinates as a list of (i, j, k).
    state is assumed to be an array of shape (num_queens, 3).
    """
    return [tuple(map(int, p)) for p in state]


# ---------------- Energy model ---------------- #

def queens_attack(p1, p2):
    """Return True if two queens attack each other in 3D."""
    i1, j1, k1 = p1
    i2, j2, k2 = p2
    di, dj, dk = i1 - i2, j1 - j2, k1 - k2

    if di == 0 and dj == 0 and dk == 0:
        return False

    # Axial attack: two coordinates are the same, the third is different
    if (di == 0 and dj == 0) or (di == 0 and dk == 0) or (dj == 0 and dk == 0):
        return True

    adi, adj, adk = abs(di), abs(dj), abs(dk)

    # Planar diagonal attacks
    if dk == 0 and adi == adj and adi != 0:
        return True
    if dj == 0 and adi == adk and adi != 0:
        return True
    if di == 0 and adj == adk and adj != 0:
        return True

    # 3D space diagonal attack
    if adi == adj == adk and adi != 0:
        return True

    return False


def compute_energy(state):
    """
    Compute number of attacking queen pairs.
    
    Args:
        state: array of shape (num_queens, 3).
    
    Returns:
        Number of attacking pairs.
    
    Complexity: O(num_queens^2) = O(N^4).
    """
    queens = get_positions(state)
    E = 0
    for idx in range(len(queens)):
        for jdx in range(idx + 1, len(queens)):
            if queens_attack(queens[idx], queens[jdx]):
                E += 1
    return E


def compute_delta_energy(state, q_idx, new_pos):
    """
    Compute ΔE when moving *one* queen to a new position.

    Args:
        state: current state, array (num_queens, 3)
        q_idx: index of the queen to move (0 .. num_queens-1)
        new_pos: tuple (i_new, j_new, k_new)

    Returns:
        ΔE = E_new - E_old
        (This function does not modify state, only computes the energy change)
    """
    queens = get_positions(state)
    old_pos = queens[q_idx]

    old_conf = 0
    new_conf = 0

    for idx, pos in enumerate(queens):
        if idx == q_idx:
            continue
        if queens_attack(old_pos, pos):
            old_conf += 1
        if queens_attack(new_pos, pos):
            new_conf += 1

    return new_conf - old_conf

