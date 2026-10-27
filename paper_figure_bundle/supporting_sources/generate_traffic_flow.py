import pandas as pd
import numpy as np
import os

# Configuration
OUTPUT_DIR = "./data_traffic/"

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
    Defines the 'Base Groups' (Scenarios) using exact edge matching.
    Returns a dictionary of scenarios: { (origin, dest): flow_rate }
    """
    od_pairs = get_network_od_pairs()
    scenarios = {}

    # --- Define Exact Boundary Groups ---
    # Based on the node mappings in the script:
    # Top (North):    np1..np5
    # Right (East):   np6..np10
    # Bottom (South): np11..np15
    # Left (West):    np16..np20
    
    # We define identifying sets for string checking
    north_nodes = {'np1', 'np2', 'np3', 'np4', 'np5'}
    east_nodes  = {'np6', 'np7', 'np8', 'np9', 'np10'}
    south_nodes = {'np11', 'np12', 'np13', 'np14', 'np15'}
    west_nodes  = {'np16', 'np17', 'np18', 'np19', 'np20'}

    # Helper to check if an edge belongs to a group
    # origin string format: "npX_ntY" -> We check X
    # dest string format:   "ntY_npX" -> We check X
    def is_edge_in_group(edge_name, group_nodes, is_origin):
        parts = edge_name.split('_')
        # If origin: npX is parts[0]. If dest: npX is parts[1]
        node_id = parts[0] if is_origin else parts[1]
        return node_id in group_nodes

    # --- Scenario 1: Uniform Low Traffic ---
    scenarios['uniform_low'] = {pair: 10.0 for pair in od_pairs}

    # --- Scenario 2: West-to-East Peak ---
    # Flow from Left(West) -> Right(East)
    scenarios['west_east_peak'] = {}
    count_we = 0
    for pair in od_pairs:
        o, d = pair
        
        # Strict Check
        valid_origin = is_edge_in_group(o, west_nodes, is_origin=True)
        valid_dest   = is_edge_in_group(d, east_nodes, is_origin=False)
        
        if valid_origin and valid_dest:
            scenarios['west_east_peak'][pair] = 120.0
            count_we += 1
        else:
            scenarios['west_east_peak'][pair] = 5.0
            
    print(f"DEBUG: West-East Peak configured with {count_we} high-flow pairs.")

    # --- Scenario 3: North-to-South Peak ---
    # Flow from Top(North) -> Bottom(South)
    scenarios['north_south_peak'] = {}
    count_ns = 0
    for pair in od_pairs:
        o, d = pair
        
        # Strict Check
        valid_origin = is_edge_in_group(o, north_nodes, is_origin=True)
        valid_dest   = is_edge_in_group(d, south_nodes, is_origin=False)
        
        if valid_origin and valid_dest:
            scenarios['north_south_peak'][pair] = 120.0
            count_ns += 1
        else:
            scenarios['north_south_peak'][pair] = 5.0

    print(f"DEBUG: North-South Peak configured with {count_ns} high-flow pairs.")

    return scenarios
def save_flow_rates_csv(scenario_name, flow_distribution, output_path):
    """
    Saves the flow rates (veh/hour) directly to CSV for each OD pair.
    No probabilistic sampling (Poisson) or timestamp generation occurs here.
    """
    data = []
    print(f"Processing scenario: {scenario_name}...")

    for (origin, dest), flow_rate in flow_distribution.items():
        # Optional: Skip 0 flow rows to keep file smaller
        if flow_rate <= 0:
            continue
            
        data.append({
            'origin_edge': origin,
            'dest_edge': dest,
            'veh_per_hour': float(flow_rate)
        })
            
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    if not df.empty:
        # Sort by origin then dest for readability
        df = df.sort_values(by=['origin_edge', 'dest_edge'])
        df.to_csv(output_path, index=False)
        print(f"-> Saved flow rates for {len(df)} OD pairs to {output_path}")
    else:
        print(f"-> Warning: No flow data for {scenario_name}")

if __name__ == "__main__":
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Get Distributions
    distributions = get_traffic_distributions()
    
    # 2. Generate CSVs (Demand Matrices)
    for name, dist in distributions.items():
        filename = os.path.join(OUTPUT_DIR, f"traffic_{name}.csv")
        save_flow_rates_csv(name, dist, filename)