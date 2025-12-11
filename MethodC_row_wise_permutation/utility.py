
import numpy as np

########################################################
# Utility functions for the N^2 Queens problem
########################################################

# state representation #

def random_state(N, rng=None):
    """
    Initialize an N x N height matrix.
    Each row is an independent random permutation of [0, N).
    """
    if rng is None:
        rng = np.random.default_rng()

    state = np.empty((N, N), dtype=int)
    base = np.arange(N)
    for i in range(N):
        state[i] = rng.permutation(base)

    return state


def get_positions(state):
    """Flatten the height matrix into queen coordinates [(i, j, k), ...]."""
    N = state.shape[0]
    positions = []
    for i in range(N):
        for j in range(N):
            positions.append((i, j, int(state[i, j])))
    return positions


# Energy model #

def queens_attack(p1, p2):
    """Return True if two queens attack each other in 3D."""
    i1, j1, k1 = p1
    i2, j2, k2 = p2
    di, dj, dk = i1 - i2, j1 - j2, k1 - k2

    if di == 0 and dj == 0 and dk == 0:
        return False

    if (di == 0 and dj == 0) or (di == 0 and dk == 0) or (dj == 0 and dk == 0):
        return True

    adi, adj, adk = abs(di), abs(dj), abs(dk)

    if dk == 0 and adi == adj and adi != 0:
        return True
    if dj == 0 and adi == adk and adi != 0:
        return True
    if di == 0 and adj == adk and adj != 0:
        return True
    if adi == adj == adk and adi != 0:
        return True
    return False


def compute_energy(state):
    """Compute number of attacking queen pairs (O((N^2)^2))."""
    queens = get_positions(state)
    E = 0
    for idx in range(len(queens)):
        for jdx in range(idx + 1, len(queens)):
            if queens_attack(queens[idx], queens[jdx]):
                E += 1
    return E


def compute_delta_energy(state, i, j1, j2):
    """
    Compute energy change when swapping two entries in the same row i:
        (i, j1, h1) <-> (i, j2, h2).

    Args:
        state: Current N x N height matrix.
        i: Row index (0-based).
        j1, j2: Column indices (0-based), j1 != j2.

    Returns:
        ΔE = E_new - E_old
    """
    N = state.shape[0]

    h1 = int(state[i, j1])
    h2 = int(state[i, j2])

    # Before swap
    old1 = (i, j1, h1)
    old2 = (i, j2, h2)
    # After swap
    new1 = (i, j1, h2)
    new2 = (i, j2, h1)

    old_conf = 0
    new_conf = 0

    # Conflicts with all other cells (except the two being swapped)
    for r in range(N):
        for c in range(N):
            if r == i and (c == j1 or c == j2):
                continue
            pos = (r, c, int(state[r, c]))

            if queens_attack(old1, pos):
                old_conf += 1
            if queens_attack(old2, pos):
                old_conf += 1

            if queens_attack(new1, pos):
                new_conf += 1
            if queens_attack(new2, pos):
                new_conf += 1

    # Conflicts between the two swapped positions themselves
    if queens_attack(old1, old2):
        old_conf += 1
    if queens_attack(new1, new2):
        new_conf += 1

    return new_conf - old_conf
