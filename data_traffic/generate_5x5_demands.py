#!/usr/bin/env python3
"""
Generate traffic demand groups for the 5x5 Grid with Realistic Noise.
Naming Convention: u_v (No 'road_' prefix).
Output: TRULY COMPLETE Edge-based OD CSVs.
        - Every CSV contains ALL potential OD pairs in the network.
        - Active group flows get Gaussian noise.
        - Inactive pairs get sparse Background Noise.
"""

import argparse
import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
import networkx as nx

# Use a non-interactive backend when no display is available.
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Configuration & Constants
# -----------------------------------------------------------------------------
ROWS = 5
COLS = 5
DEFAULT_TOTAL_FLOW = 2000

# Noise Configuration
NOISE_STD_DEV_PERCENT = 0.05  # 5% variation on active flows
BACKGROUND_NOISE_PROB = 0.25  # 25% chance a "zero" route gets traffic
BACKGROUND_NOISE_MEAN = 0.5  # Avg flow for background noise (veh/hr)
BACKGROUND_NOISE_STD = 1.0  # Std dev for background noise


def get_node_id(row: int, col: int) -> str:
    """Returns inner node ID (e.g., nt1..nt25)."""
    idx = (row - 1) * 5 + col
    return f"nt{idx}"


def get_edge_id(u: str, v: str) -> str:
    """Returns edge ID following the 'u_v' convention."""
    return f"{u}_{v}"


def parse_node_id(nid: str) -> Tuple[int, int]:
    """Parses node ID to logical (row, col)."""
    if nid.startswith("nt"):
        idx = int(nid[2:])
        row = math.ceil(idx / 5)
        col = idx - (row - 1) * 5
        return row, col
    return 0, 0


# -----------------------------------------------------------------------------
# 2. Graph Generation
# -----------------------------------------------------------------------------
def build_extended_grid() -> Tuple[nx.DiGraph, Dict[str, Tuple[float, float]], Dict[str, Dict]]:
    g = nx.DiGraph()
    pos = {}
    road_info = {}

    # A. Build Inner Grid (nt1 - nt25)
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            nid = get_node_id(r, c)
            pos[nid] = (float(c), float(r))
            g.add_node(nid)

    # B. Build Inner Edges
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            u = get_node_id(r, c)
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            for nr, nc in neighbors:
                if 1 <= nr <= ROWS and 1 <= nc <= COLS:
                    v = get_node_id(nr, nc)
                    rid = get_edge_id(u, v)
                    g.add_edge(u, v)
                    road_info[rid] = {"start": u, "end": v}

    # C. Build Peripheral Nodes (np1 - np20) & Connections
    peripherals = []
    # Bottom (np1..np5)
    for i in range(5): peripherals.append((f"np{i + 1}", 1, i + 1, float(i + 1), 0.0))
    # Right (np6..np10)
    for i in range(5): peripherals.append((f"np{i + 6}", i + 1, 5, 6.0, float(i + 1)))
    # Top (np11..np15)
    for i in range(5): peripherals.append((f"np{i + 11}", 5, 5 - i, float(5 - i), 6.0))
    # Left (np16..np20)
    for i in range(5): peripherals.append((f"np{i + 16}", 5 - i, 1, 0.0, float(5 - i)))

    for pid, r, c, px, py in peripherals:
        pos[pid] = (px, py)
        g.add_node(pid)
        target_nt = get_node_id(r, c)

        # Incoming: np -> nt (e.g., "np1_nt1")
        in_edge = get_edge_id(pid, target_nt)
        g.add_edge(pid, target_nt)
        road_info[in_edge] = {"start": pid, "end": target_nt}

        # Outgoing: nt -> np (e.g., "nt1_np1")
        out_edge = get_edge_id(target_nt, pid)
        g.add_edge(target_nt, pid)
        road_info[out_edge] = {"start": target_nt, "end": pid}

    return g, pos, road_info


# -----------------------------------------------------------------------------
# 3. Region Logic
# -----------------------------------------------------------------------------
def define_regions(g: nx.DiGraph, road_info: Dict[str, Dict]) -> Dict[str, Set[str]]:
    regions = defaultdict(set)

    # 1. Peripheral Incoming
    for rid, info in road_info.items():
        u = info["start"]
        if u.startswith("np"):
            regions["periphery_origins"].add(rid)
            idx = int(u[2:])
            if 1 <= idx <= 5:
                regions["south"].add(rid)
                if idx <= 3: regions["sw"].add(rid)
                if idx >= 3: regions["se"].add(rid)
            elif 6 <= idx <= 10:
                regions["east"].add(rid)
                if idx <= 8: regions["se"].add(rid)
                if idx >= 8: regions["ne"].add(rid)
            elif 11 <= idx <= 15:
                regions["north"].add(rid)
                if idx <= 13: regions["ne"].add(rid)
                if idx >= 13: regions["nw"].add(rid)
            elif 16 <= idx <= 20:
                regions["west"].add(rid)
                if idx <= 18: regions["nw"].add(rid)
                if idx >= 18: regions["sw"].add(rid)

    # 2. Peripheral Outgoing
    for rid, info in road_info.items():
        if info["end"].startswith("np"):
            regions["periphery_dests"].add(rid)

    # 3. Center
    center_indices = {7, 8, 9, 12, 13, 14, 17, 18, 19}
    center_nodes = {f"nt{i}" for i in center_indices}

    for rid, info in road_info.items():
        u, v = info["start"], info["end"]
        if u.startswith("np") or v.startswith("np"): continue

        if u in center_nodes: regions["center_origins"].add(rid)
        if v in center_nodes: regions["center_dests"].add(rid)

    return regions


# -----------------------------------------------------------------------------
# 4. Path & Demand Logic
# -----------------------------------------------------------------------------
def precompute_shortest_paths(g: nx.DiGraph, road_info: Dict[str, Dict]) -> Dict[Tuple[str, str], List[List[str]]]:
    path_edges = defaultdict(list)
    sources = list(g.nodes)
    targets = list(g.nodes)

    print("Precomputing shortest paths...")
    for o in sources:
        for d in targets:
            if o == d: continue
            try:
                for node_path in nx.all_shortest_paths(g, o, d):
                    edge_path = []
                    valid = True
                    for i in range(len(node_path) - 1):
                        u, v = node_path[i], node_path[i + 1]
                        eid = get_edge_id(u, v)
                        if eid not in road_info:
                            valid = False;
                            break
                        edge_path.append(eid)
                    if valid:
                        path_edges[(o, d)].append(edge_path)
            except nx.NetworkXNoPath:
                continue
    return path_edges


def generate_edge_od_matrix(origin_edges: Set[str], dest_edges: Set[str], total_flow: float, road_info: Dict) -> List[
    Tuple[str, str, float]]:
    """
    Generates BASE flow for valid pairs in the active group.
    Does NOT add noise yet (noise added at CSV writing for consistency).
    """
    pairs = []

    for o_edge in origin_edges:
        o_node = road_info[o_edge]["start"]
        for d_edge in dest_edges:
            d_node = road_info[d_edge]["end"]
            if o_node == d_node: continue
            pairs.append((o_edge, d_edge))

    if not pairs: return []

    base_flow = total_flow / len(pairs)
    return [(o, d, base_flow) for o, d in pairs]


def calculate_usage_and_active_flows(
        edge_od: List[Tuple[str, str, float]],
        path_map: Dict[Tuple[str, str], List[List[str]]],
        road_info: Dict
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    road_usage = defaultdict(float)
    active_flows = defaultdict(float)  # Map (o,d) -> flow

    for o_edge, d_edge, base_vol in edge_od:
        # Add Gaussian Noise to the ACTIVE flow here
        noisy_vol = max(0.0, random.gauss(base_vol, base_vol * NOISE_STD_DEV_PERCENT))
        active_flows[(o_edge, d_edge)] += noisy_vol

        # Viz Calculation
        start_node = road_info[o_edge]["start"]
        end_node = road_info[d_edge]["end"]

        paths = path_map.get((start_node, end_node))
        if not paths: continue

        valid_paths = [p for p in paths if p[-1] == d_edge]
        if not valid_paths: valid_paths = paths

        share = noisy_vol / len(valid_paths)
        for path in valid_paths:
            for edge_id in path:
                road_usage[edge_id] += share

    return road_usage, active_flows


# -----------------------------------------------------------------------------
# 5. Visualization & Main
# -----------------------------------------------------------------------------
def draw_group(ax, g, pos, road_info, road_usage, title, draw_arrows=False):
    road_ids = list(road_info.keys())
    vals = [road_usage.get(rid, 0.0) for rid in road_ids]
    max_flow = max(vals) if vals else 1.0

    edges = []
    colors = []
    widths = []

    for rid in road_ids:
        r = road_info[rid]
        u, v = r["start"], r["end"]
        flow = road_usage.get(rid, 0.0)

        if flow > 0:
            norm = flow / max_flow
            rgba = plt.cm.autumn_r(norm)
            color = (rgba[0], rgba[1], rgba[2], 0.95)
            w = 0.35 + 1.25 * norm
        else:
            color = (0.7, 0.7, 0.7, 0.25)
            w = 0.15

        edges.append((u, v))
        colors.append(color)
        widths.append(w)

    if draw_arrows:
        nx.draw_networkx_edges(
            g, pos, ax=ax, edgelist=edges, edge_color=colors, width=widths,
            arrows=True, arrowstyle='-|>', arrowsize=12, connectionstyle="arc3,rad=0.0",
            min_source_margin=2, min_target_margin=2
        )
    else:
        # Faster path: LineCollection render avoids expensive FancyArrowPatch objects.
        nx.draw_networkx_edges(
            g, pos, ax=ax, edgelist=edges, edge_color=colors, width=widths, arrows=False
        )

    nt_nodes = [n for n in g.nodes if n.startswith("nt")]
    np_nodes = [n for n in g.nodes if n.startswith("np")]
    nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=nt_nodes, node_size=10, node_color="#555555")
    nx.draw_networkx_nodes(g, pos, ax=ax, nodelist=np_nodes, node_size=20, node_color="#0000ff")

    ax.set_title(title, fontsize=16, pad=8)
    ax.axis("off")
    ax.set_aspect('equal')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("data_traffic"))
    parser.add_argument("--flow", type=float, default=DEFAULT_TOTAL_FLOW)
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open a blocking figure window after saving.")
    parser.add_argument("--dpi", type=int, default=180,
                        help="PNG resolution. Lower values save faster.")
    parser.add_argument("--tight-bbox", action="store_true",
                        help="Use bbox_inches='tight' (can be slower on complex plots).")
    parser.add_argument("--draw-arrows", action="store_true",
                        help="Draw directional arrowheads (slower to render/save).")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    g, pos, road_info = build_extended_grid()
    regions = define_regions(g, road_info)
    path_map = precompute_shortest_paths(g, road_info)

    p_in = regions["periphery_origins"]
    p_out = regions["periphery_dests"]
    c_in = regions["center_dests"]
    c_out = regions["center_origins"]

    # --- BUILD UNIVERSAL SETS ---
    # We want every CSV to contain rows for ALL potential Origin/Dest edges
    # used in any scenario.
    all_possible_origins = set().union(p_in, c_out)
    all_possible_dests = set().union(p_out, c_in)

    def get_p_out_subset(indices):
        subset = set()
        for rid in p_out:
            end_node = road_info[rid]["end"]
            idx = int(end_node[2:])
            if idx in indices: subset.add(rid)
        return subset

    idx_s = set(range(1, 6))
    idx_e = set(range(6, 11))
    idx_n = set(range(11, 16))
    idx_w = set(range(16, 21))

    group_specs = [
        ("N->S", regions["north"], get_p_out_subset(idx_s)),
        ("S->N", regions["south"], get_p_out_subset(idx_n)),
        ("W->E", regions["west"], get_p_out_subset(idx_e)),
        ("E->W", regions["east"], get_p_out_subset(idx_w)),
        ("NW->SE", regions["nw"], get_p_out_subset({3, 4, 5, 6, 7, 8})),
        ("SE->NW", regions["se"], get_p_out_subset({13, 14, 15, 16, 17, 18})),
        ("SW->NE", regions["sw"], get_p_out_subset({8, 9, 10, 11, 12, 13})),
        ("NE->SW", regions["ne"], get_p_out_subset({18, 19, 20, 1, 2, 3})),
        ("Periphery->Center", p_in, c_in),
        ("Center->Periphery", c_out, p_out),
        ("Uniform", p_in, p_out),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    print(f"\nGenerating Realistic Demands (Total Flow: {args.flow})...")
    print(f"Noise: {int(NOISE_STD_DEV_PERCENT * 100)}% on active flows.")
    print(f"Background: {int(BACKGROUND_NOISE_PROB * 100)}% chance of {int(BACKGROUND_NOISE_MEAN)} vph noise.")

    for i, (name, origins, dests) in enumerate(group_specs):
        safe_name = name.replace("->", "_to_")

        # 1. Generate Base Flows for this group
        edge_od_base = generate_edge_od_matrix(origins, dests, args.flow, road_info)

        # 2. Add Gaussian Noise + Calc Viz
        usage, active_flows = calculate_usage_and_active_flows(edge_od_base, path_map, road_info)

        csv_path = args.outdir / f"demand_{safe_name}.csv"

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["origin_edge", "dest_edge", "veh_per_hour"])

            # 3. Iterate over UNIVERSAL Sets (Completeness)
            sorted_all_o = sorted(list(all_possible_origins))
            sorted_all_d = sorted(list(all_possible_dests))

            for o in sorted_all_o:
                for d in sorted_all_d:
                    # Sanity check: Start Node != End Node
                    u = road_info[o]["start"]
                    v = road_info[d]["end"]
                    if u == v: continue

                    # Check if this pair is active in current group
                    flow = active_flows.get((o, d), 0.0)

                    # 4. Add Background Noise to Inactive pairs
                    if flow == 0.0:
                        if random.random() < BACKGROUND_NOISE_PROB:
                            # Sparse random noise
                            flow = random.gauss(BACKGROUND_NOISE_MEAN, BACKGROUND_NOISE_STD)
                            flow = max(0.0, flow)

                    writer.writerow([o, d, f"{flow:.4f}"])

        draw_group(axes[i], g, pos, road_info, usage, name, draw_arrows=args.draw_arrows)
        print(f"  - {name}: Saved {csv_path.name}")

    for j in range(len(group_specs), len(axes)): axes[j].axis("off")

    plt.tight_layout()
    fig_path = args.outdir / "final_demand_patterns.png"
    save_kwargs = {"dpi": args.dpi}
    if args.tight_bbox:
        save_kwargs["bbox_inches"] = "tight"

    print(f"Saving PNG to {fig_path} ...")
    fig.savefig(fig_path, **save_kwargs)
    print("PNG saved.")

    if not args.no_show:
        print("Displaying plot window (close it to finish).")
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
