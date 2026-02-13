import pandas as pd
import numpy as np
import os

# Configuration
OUTPUT_FILE = "traffic_flow_5x5_high.csv"
DURATION_SEC = 3600  # 1 hour simulation
LAMBDA_FLOW = 2    # Poisson parameter (approx vehicles per second)
SPEED_LIMIT = 11.0   # Matches SPEED_LIMIT_AV in build_file.py

def get_network_edges():
    """
    Reconstructs the valid external edges based on large_grid/data/build_file.py logic.
    Returns lists of valid origin_edges and dest_edges.
    """
    # Mapping from build_file.py logic:
    # Group 1
    nt_indices_1 = [5, 10, 15, 20, 25, 21, 16, 11, 6, 1]
    np_indices_1 = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
    
    # Group 2
    nt_indices_2 = [1, 2, 3, 4, 5, 25, 24, 23, 22, 21]
    np_indices_2 = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]

    origins = []
    destinations = []

    # Process Group 1
    for nt_i, np_i in zip(nt_indices_1, np_indices_1):
        # Incoming: From Periphery (np) -> Network (nt)
        origins.append(f"np{np_i}_nt{nt_i}")
        # Outgoing: From Network (nt) -> Periphery (np)
        destinations.append(f"nt{nt_i}_np{np_i}")

    # Process Group 2
    for nt_i, np_i in zip(nt_indices_2, np_indices_2):
        origins.append(f"np{np_i}_nt{nt_i}")
        destinations.append(f"nt{nt_i}_np{np_i}")

    return sorted(list(set(origins))), sorted(list(set(destinations)))

def generate_traffic_data():
    origins, destinations = get_network_edges()
    print(f"Found {len(origins)} origins and {len(destinations)} destinations.")

    data = []
    
    # Generate traffic step-by-step
    for sec in range(DURATION_SEC):
        # Determine number of vehicles to spawn this second (Poisson distribution)
        num_vehicles = np.random.poisson(LAMBDA_FLOW)
        
        for _ in range(num_vehicles):
            # Randomly pick origin and destination
            origin = np.random.choice(origins)
            dest = np.random.choice(destinations)
            
            # Simple check to avoid immediate U-turns at the same node pair
            # (e.g. np6_nt5 -> nt5_np6)
            # Extract numbers to check
            origin_parts = origin.split('_')
            dest_parts = dest.split('_')
            if origin_parts[0] == dest_parts[1] and origin_parts[1] == dest_parts[0]:
                continue # Skip this pair and try next time
            
            data.append({
                'depart_sec': sec,
                'origin_edge': origin,
                'dest_edge': dest,
                'speed': SPEED_LIMIT
            })

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Excel
    print(f"Generated {len(df)} vehicles. Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    generate_traffic_data()