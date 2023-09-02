import h3
import json
from collections import deque

class H3GlobalIterator:
    """
    Represents the linear iteration over all H3 cells at a given resolution, starting
    at a seed resolution. Provides the ability to save and restore state.
    """
    def __init__(self, seed_lat, seed_lng, resolution, state_file='h3_state.json'):
        self.resolution = resolution
        self.state_file = state_file
        
        self.seen = set()
        self.queue = deque()
        seed = h3.geo_to_h3(seed_lat, seed_lng, resolution)
        self.queue.append(seed)
        self.seen.add(seed)
            
        if state_file:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.seen = set(state['seen'])
                self.queue = deque(state['queue'])


    def save_state(self, state_filepath):
        with open(state_filepath, 'w') as f:
            state = {'seen': list(self.seen), 'queue': list(self.queue)}
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


def h3_global_iterator(seed_lat, seed_lng, resolution):
    """
    A generator to iterate over all h3 cells globally at a given resolution.

    Args:
    - resolution (int): The resolution of the H3 index.

    Yields:
    - str: An H3 index.
    """
    
    seen = set()  # Track cells we've already seen.
    queue = deque()  # For BFS traversal.
    
    # Seed cell at coordinates (0,0).
    seed = h3.geo_to_h3(seed_lat, seed_lng, resolution)
    queue.append(seed)
    seen.add(seed)
    
    while queue:
        current_cell = queue.popleft()
        yield current_cell
        
        # Get the neighbors of the current cell.
        neighbors = h3.k_ring(current_cell, 1)
        
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)