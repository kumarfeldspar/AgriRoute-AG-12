#!/usr/bin/env python
"""
baseline.py
-----------
A naive/baseline approach:
  - Ignores capacity constraints.
  - Assigns each farm to the closest hub, then each hub to the closest center.
  - Uses a single 'average' cost for transport, ignoring vehicle differences.
Outputs:
  baseline_solution.json
"""

import json
import math

def distance(lat1, lon1, lat2, lon2):
    # Haversine or Euclidean approximation:
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def main():
    with open("farms.json") as f:
        farms = json.load(f)
    with open("hubs.json") as f:
        hubs = json.load(f)
    with open("centers.json") as f:
        centers = json.load(f)
    with open("disruptions.json") as f:
        disruptions = json.load(f)

    # Convert disruptions to a quick dictionary for cost multiplier / closures
    disruption_map = {}
    for d in disruptions:
        start = d["start"]  # e.g. ("farm", 0)
        end = d["end"]      # e.g. ("hub", 1)
        disruption_map[(start[0], start[1], end[0], end[1])] = {
            "closed": d["closed"],
            "cost_mult": d["cost_multiplier"]
        }

    solution = {
        "stage1_routes": [],  # farm -> hub
        "stage2_routes": [],  # hub -> center
        "total_cost": 0.0,
        "total_spoilage": 0.0
    }

    # Baseline cost parameters
    base_cost_per_dist = 1.0

    # 1) Assign each farm to nearest hub
    stage1_cost = 0
    for farm in farms:
        fx, fy = farm["location"][0], farm["location"][1]
        best_hub = None
        best_dist = float("inf")
        for hub in hubs:
            hx, hy = hub["location"][0], hub["location"][1]
            dist = distance(fx, fy, hx, hy)
            # Check if this link is closed
            disrupt = disruption_map.get(("farm", farm["id"], "hub", hub["id"]), None)
            if disrupt and disrupt["closed"]:
                # road closed, skip
                continue
            if dist < best_dist:
                best_dist = dist
                best_hub = hub

        if best_hub is not None:
            # check cost multiplier
            disrupt = disruption_map.get(("farm", farm["id"], "hub", best_hub["id"]), {"cost_mult": 1.0})
            cost = base_cost_per_dist * best_dist * disrupt["cost_mult"]
            # naive spoilage check: if best_dist is too large or if farm's perish_window < some threshold
            # We'll do a simplistic approach
            # If best_dist > farm["perish_window"], half produce spoils
            # (completely naive example)
            if best_dist > farm["perish_window"]:
                spoil = farm["quantity"] * 0.5
            else:
                spoil = 0

            solution["stage1_routes"].append({
                "farm_id": farm["id"],
                "hub_id": best_hub["id"],
                "distance": best_dist,
                "cost": cost,
                "spoilage": spoil
            })
            stage1_cost += cost
        else:
            # If no open route to any hub, all produce spoils
            solution["stage1_routes"].append({
                "farm_id": farm["id"],
                "hub_id": None,
                "distance": None,
                "cost": 0,
                "spoilage": farm["quantity"]
            })

    # 2) Assign each hub to the nearest center
    stage2_cost = 0
    for hub in hubs:
        hx, hy = hub["location"][0], hub["location"][1]
        best_center = None
        best_dist = float("inf")
        for center in centers:
            cx, cy = center["location"][0], center["location"][1]
            dist = distance(hx, hy, cx, cy)
            # check closure
            disrupt = disruption_map.get(("hub", hub["id"], "center", center["id"]), None)
            if disrupt and disrupt["closed"]:
                continue
            if dist < best_dist:
                best_dist = dist
                best_center = center

        if best_center is not None:
            disrupt = disruption_map.get(("hub", hub["id"], "center", best_center["id"]), {"cost_mult": 1.0})
            cost = base_cost_per_dist * best_dist * disrupt["cost_mult"]
            # no sophisticated spoilage, just do minimal
            # if dist > 5 => possible spoilage
            spoil = 0

            solution["stage2_routes"].append({
                "hub_id": hub["id"],
                "center_id": best_center["id"],
                "distance": best_dist,
                "cost": cost,
                "spoilage": spoil
            })
            stage2_cost += cost
        else:
            solution["stage2_routes"].append({
                "hub_id": hub["id"],
                "center_id": None,
                "distance": None,
                "cost": 0,
                "spoilage": 0
            })

    solution["total_cost"] = round(stage1_cost + stage2_cost, 2)
    # sum spoilage
    total_spoil = 0
    for r in solution["stage1_routes"]:
        total_spoil += r["spoilage"]
    for r in solution["stage2_routes"]:
        total_spoil += r["spoilage"]
    solution["total_spoilage"] = total_spoil

    # Save
    with open("baseline_solution.json", "w") as f:
        json.dump(solution, f, indent=4)
    print("Baseline solution created: baseline_solution.json")

if __name__ == "__main__":
    main()
