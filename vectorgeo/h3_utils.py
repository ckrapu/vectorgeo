import h3
import json
from collections import deque


class H3GlobalIterator:
    """
    Represents the linear iteration over all H3 cells at a given resolution, starting
    at a seed resolution. Provides the ability to save and restore state.
    """

    def __init__(self, seed_lat, seed_lng, resolution, state_file="h3_state.json"):
        self.resolution = resolution
        self.state_file = state_file

        self.seen = set()
        self.queue = deque()
        seed = h3.geo_to_h3(seed_lat, seed_lng, resolution)
        self.queue.append(seed)
        self.seen.add(seed)

        if state_file:
            with open(self.state_file, "r") as f:
                state = json.load(f)
                self.seen = set(state["seen"])
                self.queue = deque(state["queue"])

    def save_state(self, state_filepath):
        with open(state_filepath, "w") as f:
            state = {"seen": list(self.seen), "queue": list(self.queue)}
            json.dump(state, f)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.queue:
            raise StopIteration
        current_cell = self.queue.popleft()
        neighbors = h3.k_ring(current_cell, 1)
        for neighbor in neighbors:
            if neighbor not in self.seen:
                self.seen.add(neighbor)
                self.queue.append(neighbor)
        return current_cell


def generate_h3_indexes_at_resolution(resolution):
    all_h3_indexes = set()

    # Start with the 122 base cells
    base_cells = h3.get_res0_indexes()

    # Recursive function to refine cells
    def refine_cell(cell, current_res):
        nonlocal all_h3_indexes
        if current_res == resolution:
            all_h3_indexes.add(cell)
            return
        children = h3.uncompact([cell], current_res + 1)
        for child in children:
            refine_cell(child, current_res + 1)

    # Start the recursion
    for base_cell in base_cells:
        refine_cell(base_cell, 0)

    return all_h3_indexes
