
import numpy as np

########################################################
# Utility functions for the N^2 Queens problem
########################################################


# state representation #

def random_state(N, rng=None):
    """
    Each column (i, j) stores a height in [0, N), so the state is an integer N x N matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, N, size=(N, N))


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


def compute_delta_energy(state, i, j, old_h, new_h):
    """Compute energy change when changing position (i, j) from old_h to new_h (O(N^2))."""
    N = state.shape[0]
    old_pos = (i, j, old_h)
    new_pos = (i, j, new_h)
    
    # Count attacks involving the old position (to subtract)
    old_attacks = 0
    # Count attacks involving the new position (to add)
    new_attacks = 0
    
    for i2 in range(N):
        for j2 in range(N):
            if i2 == i and j2 == j:
                continue  # Skip the position being changed
            pos2 = (i2, j2, int(state[i2, j2]))
            if queens_attack(old_pos, pos2):
                old_attacks += 1
            if queens_attack(new_pos, pos2):
                new_attacks += 1
    
    return new_attacks - old_attacks
