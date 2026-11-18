"""
PathSnap Sequential Route Reconstruction (HMM & Viterbi)
Module for Member 3: Sequential state decoding, emission & transition costs, and Viterbi trellis.
"""

import math
from typing import List, Dict, Tuple
import numpy as np


class HMMRouteDecoder:
    """
    Hidden Markov Model decoder for map-matching:
    - States: Candidate road segments (arcs)
    - Observations: Noisy GPS fixes
    - Emission Probability: P(z_t | r_i) based on perpendicular distance
    - Transition Probability: P(r_j | r_i) based on shortest road distance vs Euclidean step
    """

    def __init__(self, sigma_z: float = 10.0, beta: float = 5.0):
        self.sigma_z = sigma_z  # GPS measurement noise standard deviation (meters)
        self.beta = beta        # Exponential transition discrepancy parameter

    def compute_emission_prob(self, distance_meters: float) -> float:
        """Gaussian observation emission probability density."""
        coeff = 1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_z)
        return coeff * math.exp(-0.5 * (distance_meters / self.sigma_z) ** 2)

    def compute_transition_prob(self, road_distance: float, great_circle_distance: float) -> float:
        """Exponential transition probability penalizing detour discrepancy."""
        delta = abs(road_distance - great_circle_distance)
        return (1.0 / self.beta) * math.exp(-delta / self.beta)

    def viterbi_decode(self, emission_matrix: np.ndarray, transition_matrix: np.ndarray) -> List[int]:
        """
        Viterbi dynamic programming decoder in log-space.
        Returns the most probable sequence of state indices.
        """
        T, N = emission_matrix.shape
        log_V = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)

        # Initial state probabilities
        log_V[0] = np.log(emission_matrix[0] + 1e-12)

        # Forward recursion
        for t in range(1, T):
            for j in range(N):
                log_trans = np.log(transition_matrix[:, j] + 1e-12)
                scores = log_V[t - 1] + log_trans
                best_prev = np.argmax(scores)
                backpointer[t, j] = best_prev
                log_V[t, j] = scores[best_prev] + np.log(emission_matrix[t, j] + 1e-12)

        # Backtrack optimal path
        best_path = [int(np.argmax(log_V[-1]))]
        for t in range(T - 1, 0, -1):
            best_path.append(int(backpointer[t, best_path[-1]]))

        return best_path[::-1]
