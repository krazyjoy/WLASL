import numpy as np


pose_edges = [
    (0, 1), (1, 2), (2, 3), (3, 7),         # Right arm
    (0, 4), (4, 5), (5, 6), (6, 8),         # Left arm
    (9, 10),                               # Shoulders
    (11, 12), (11, 13), (13, 15), (15, 17), # Left leg
    (12, 14), (14, 16), (16, 18),           # Right leg
    (11, 23), (12, 24),                     # Torso to legs
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), # Legs to feet
]


hand_edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# LH starts at index 33
LH_OFFSET = 33
left_hand_edges = [(a + LH_OFFSET, b + LH_OFFSET) for a, b in hand_edges]

# RH starts at index 54
RH_OFFSET = 54
right_hand_edges = [(a + RH_OFFSET, b + RH_OFFSET) for a, b in hand_edges]

LEFT_WRIST = 15
RIGHT_WRIST = 16

hand_to_pose = [
    (LEFT_WRIST, LH_OFFSET),
    (RIGHT_WRIST, RH_OFFSET)
]


def build_mediapipe_edge_list():
    edge_list = []
    edge_list += pose_edges
    edge_list += left_hand_edges
    edge_list += right_hand_edges
    edge_list += hand_to_pose
    return edge_list


def build_adjacency_matrix(num_nodes, edges):
    adj = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    np.fill_diagonal(adj, 1)
    return adj



