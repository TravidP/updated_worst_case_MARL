import pandas as pd
import numpy as np
import os

# Configuration
OUTPUT_DIR = "./data_traffic/"
SPEED_LIMIT = 11.0   # Matches standard large grid speed
DURATION_SEC = 3600  # 1 hour simulation

def get_network_od_pairs():
    """
    Reconstructs the valid external edges based on the 5x5 Large Grid topology.
    Returns a list of tuples: [(origin_edge, dest_edge), ...]
    """
    # Topology mappings from original large_grid configuration
    # Group 1 (Vertical sides)
    nt_indices_1 = [5, 10, 15, 20, 25, 21, 16, 11, 6, 1]
    np_indices_1 = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
    
    # Group 2 (Horizontal sides)
    nt_indices_2 = [1, 2, 3, 4, 5, 25, 24, 23, 22, 21]
    np_indices_2 = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]

    origins = []
    destinations = []

    # Helper to build edge names
    # Origin format: np{i}_nt{j}
    # Dest format:   nt{j}_np{i}
    
    for nt_i, np_i in zip(nt_indices_1, np_indices_1):
        origins.append(f"np{np_i}_nt{nt_i}")
        destinations.append(f"nt{nt_i}_np{np_i}")

    for nt_i, np_i in zip(nt_indices_2, np_indices_2):
        origins.append(f"np{np_i}_nt{nt_i}")
        destinations.append(f"nt{nt_i}_np{np_i}")

    # Generate all permutations (All-to-All)
    valid_od_pairs = []
    for o in origins:
        for d in destinations:
            # Filter: Don't allow U-turns at the exact same node boundary
            # e.g., np6_nt5 -> nt5_np6 is an immediate U-turn
            o_parts = o.split('_') # ['np6', 'nt5']
            d_parts = d.split('_') # ['nt5', 'np6']
            
            # If the numbers match reversed, it's a U-turn at entry
            if o_parts[0] == d_parts[1] and o_parts[1] == d_parts[0]:
                continue
                
            valid_od_pairs.append((o, d))
            
    return valid_od_pairs

def get_traffic_distributions():
    """
    Defines the 'Base Groups' (Scenarios).
    Returns a dictionary of scenarios, where each scenario is:
    { (origin, dest): flow_rate_veh_per_hour }
    """
    od_pairs = get_network_od_pairs()
    scenarios = {}

    # --- Scenario 1: Uniform Low Traffic ---
    # Every pair gets small constant flow
    scenarios['uniform_low'] = {pair: 10.0 for pair in od_pairs} # 10 veh/hr per pair

    # --- Scenario 2: West-to-East Peak ---
    # Simulate heavy flow entering from West (Left side of grid) going East
    scenarios['west_east_peak'] = {}
    for pair in od_pairs:
        o, d = pair
        # Heuristic: edges starting with 'np6', 'np11', 'np16' are on the West side
        is_west_origin = any(x in o for x in ['np6', 'np11', 'np16', 'np1', 'np21']) 
        is_east_dest = any(x in d for x in ['np10', 'np15', 'np20', 'np5', 'np25'])
        
        if is_west_origin and is_east_dest:
            scenarios['west_east_peak'][pair] = 120.0 # High flow
        else:
            scenarios['west_east_peak'][pair] = 5.0   # Background noise

    # --- Scenario 3: North-to-South Peak ---
    scenarios['north_south_peak'] = {}
    for pair in od_pairs:
        o, d = pair
        # Heuristic for Top/Bottom edges
        is_north_origin = any(x in o for x in ['np1', 'np2', 'np3', 'np4', 'np5'])
        is_south_dest = any(x in d for x in ['np21', 'np22', 'np23', 'np24', 'np25'])
        
        if is_north_origin and is_south_dest:
            scenarios['north_south_peak'][pair] = 120.0
        else:
            scenarios['north_south_peak'][pair] = 5.0

    return scenarios

def generate_traffic_csv(scenario_name, flow_distribution, output_path, seed=42):
    """
    Generates a CSV file based on flow rates (veh/hour).
    Uses Poisson arrival process.
    """
    np.random.seed(seed)
    vehicles = []

    print(f"Generating scenario: {scenario_name}...")
    
    for (origin, dest), flow_rate_per_hr in flow_distribution.items():
        if flow_rate_per_hr <= 0:
            continue
            
        # 1. Calculate expected number of vehicles for the duration
        # Lambda = (Veh/Hour) / 3600 * Duration
        expected_n = (flow_rate_per_hr / 3600.0) * DURATION_SEC
        
        # 2. Sample actual number from Poisson
        num_veh = np.random.poisson(expected_n)
        
        if num_veh == 0:
            continue
            
        # 3. Generate departure times
        # Uniformly distributed over the hour, then sorted
        depart_times = np.sort(np.random.uniform(0, DURATION_SEC, num_veh))
        
        for t in depart_times:
            vehicles.append({
                'depart_sec': int(t),
                'origin_edge': origin,
                'dest_edge': dest,
                'speed': SPEED_LIMIT
            })
            
    # Convert to DataFrame
    df = pd.DataFrame(vehicles)
    
    if not df.empty:
        # Sort by departure time to ensure SUMO compatibility
        df = df.sort_values(by='depart_sec')
        df.to_csv(output_path, index=False)
        print(f"-> Saved {len(df)} vehicles to {output_path}")
    else:
        print(f"-> Warning: No vehicles generated for {scenario_name}")

if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Get Distributions
    distributions = get_traffic_distributions()
    
    # 2. Generate CSVs for testing
    for name, dist in distributions.items():
        filename = os.path.join(OUTPUT_DIR, f"traffic_{name}.csv")
        generate_traffic_csv(name, dist, filename)