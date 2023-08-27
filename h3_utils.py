import h3

from collections import deque

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