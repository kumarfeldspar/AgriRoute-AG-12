#!/usr/bin/env python
"""
greedy_kmeans.py
----------------
Implements a Greedy + K-Means approach (or nearest-neighbor approach) as a benchmark.

Process:
1) Cluster farms by location (K-Means) or do a simple nearest-hub grouping.
2) For each cluster, pick the smallest vehicle that can carry that cluster's total load.
3) Compute route distance using the distance matrix. Attempt to place produce in a storage hub
   if it doesn't exceed capacity. Otherwise, it may cause partial spoilage or no assignment.
4) Output a "greedy_solution.json"-like structure with cost breakdown, delivered quantities, etc.

Note: This is a simplified demonstration ignoring full complexity (like multiple trips).
"""

import json
import math
import random

import numpy as np
from sklearn.cluster import KMeans

def compute_storage_cost(units, hub_info, time_in_storage=1.0):
    """
    Simple storage cost: (units * storage_cost_per_unit * time_in_storage) + fixed_usage if used.
    """
    if units > 0:
        return units * hub_info["storage_cost_per_unit"] * time_in_storage + hub_info["fixed_usage_cost"]
    else:
        return 0.0

def run_greedy_kmeans(farms, hubs, centers, vehicles, dist_dict, output_file="greedy_solution.json"):
    """
    Returns a dictionary describing the solution, also writes JSON to output_file.
    'dist_dict' is a dictionary from distance_azure.py

    Steps:
      - Perform a simple K-means on farm locations to group them (k=#hubs or user-chosen).
      - Assign each cluster to a single hub if capacity allows.
      - For each cluster's total produce, pick a vehicle (small or large) that can handle it in one go.
      - Then route from that hub to the nearest distribution center that can fulfill the demand.
      - Compute cost: transport + storage + spoilage.
    """
    solution = {
        "method": "Greedy-KMeans",
        "clusters": [],
        "stage1_routes": [],  # farm->hub
        "stage2_routes": [],  # hub->center
        "total_cost": 0.0,
        "total_spoilage": 0.0,
        "total_delivered_on_time": 0.0,
    }

    if not farms or not hubs or not centers:
        with open(output_file, "w") as f:
            json.dump(solution, f, indent=4)
        return solution

    # 1) Prepare farm coordinate array for K-Means
    farm_coords = np.array([f["location"] for f in farms])
    k = min(len(hubs), len(farms))  # or user-defined
    kmeans = KMeans(n_clusters=k, random_state=42).fit(farm_coords)
    labels = kmeans.labels_

    # Build clusters
    clusters = {}
    for i, lab in enumerate(labels):
        if lab not in clusters:
            clusters[lab] = []
        clusters[lab].append(farms[i])

    # 2) For each cluster, pick 1 hub (the nearest hub to the cluster centroid)
    #    Then choose a vehicle that can transport the sum of produce in one trip (if possible).
    total_cost = 0.0
    total_spoilage = 0.0
    total_delivered = 0.0

    for clab, flist in clusters.items():
        cluster_info = {}
        cluster_farms = []
        total_cluster_produce = sum(f["produce_quantity"] for f in flist)
        # centroid
        centroid = np.mean([f["location"] for f in flist], axis=0)
        cluster_info["centroid"] = centroid.tolist()

        # find nearest hub to the centroid
        best_hub = None
        best_dist = float("inf")
        for h in hubs:
            hd = dist_dict.get(("farm", -999, "hub", h["id"]), None)
            # We'll do a quick hack to measure distance from centroid->hub using Haversine
            # or approximate by hooking into dist_dict with a mock "farm" -999. Instead, let's compute ourselves:
            hub_lat, hub_lon = h["location"]
            d = math.sqrt((centroid[0] - hub_lat)**2 + (centroid[1] - hub_lon)**2)
            if d < best_dist:
                best_dist = d
                best_hub = h
        if best_hub is None:
            # can't find a hub, everything spoils
            for ff in flist:
                cluster_farms.append({
                    "farm_id": ff["id"],
                    "assigned_hub": None,
                    "quantity": ff["produce_quantity"],
                    "spoilage": ff["produce_quantity"]
                })
                total_spoilage += ff["produce_quantity"]
            cluster_info["hub_id"] = None
            cluster_info["vehicle_id"] = None
            cluster_info["cluster_spoilage"] = total_spoilage
            solution["clusters"].append(cluster_info)
            continue

        # 3) Check if best_hub has enough capacity
        if total_cluster_produce > best_hub["capacity"]:
            # assume we can't store all => partial spoil
            # for a real approach, you might do partial usage or multiple hubs
            # we'll do a naive approach:
            portion_stored = best_hub["capacity"]
            portion_spoiled = total_cluster_produce - portion_stored
        else:
            portion_stored = total_cluster_produce
            portion_spoiled = 0.0

        total_spoilage += portion_spoiled

        # 4) Pick a vehicle that can carry portion_stored in one go
        #    smallest vehicle that can do so
        chosen_vehicle = None
        for v in sorted(vehicles, key=lambda x: x["capacity"]):
            if v["capacity"] >= portion_stored:
                chosen_vehicle = v
                break
        # if none found, it all spoils
        if chosen_vehicle is None:
            chosen_vehicle = {"id": None, "type": "none", "capacity": 0, "fixed_cost": 0, "variable_cost_per_distance": 0}
            portion_spoiled += portion_stored
            portion_stored = 0.0
            total_spoilage += portion_stored

        # 5) Compute distance cost from each farm to hub
        stage1_cost = 0.0
        for ff in flist:
            d_fh = dist_dict.get(("farm", ff["id"], "hub", best_hub["id"]), 0)
            # fix: if None, treat as 0 or skip
            if d_fh is None:
                # route is not possible => spoil
                stage1_cost += 0
                portion_spoiled += ff["produce_quantity"]
                total_spoilage += ff["produce_quantity"]
                assigned = 0
                sp = ff["produce_quantity"]
            else:
                assigned = ff["produce_quantity"]
                sp = 0
                cost_this_farm = (chosen_vehicle["fixed_cost"]
                                  + chosen_vehicle["variable_cost_per_distance"] * d_fh)
                # We'll distribute the fixed cost among all farms in the cluster naively
                cost_this_farm /= len(flist)
                stage1_cost += cost_this_farm

            cluster_farms.append({
                "farm_id": ff["id"],
                "assigned_hub": best_hub["id"],
                "quantity": assigned,
                "spoilage": sp
            })

        # cost for stage1 (transport)
        total_cost += stage1_cost

        # storage cost
        # assume we store the portion_stored for 1 time unit
        st_cost = compute_storage_cost(portion_stored, best_hub, time_in_storage=1.0)
        total_cost += st_cost

        # now stage2: from hub to the nearest center that can accept portion_stored
        best_center = None
        best_center_dist = float("inf")
        for c in centers:
            if c["demand"] > 0:  # can accept produce
                # distance
                d_hc = dist_dict.get(("hub", best_hub["id"], "center", c["id"]), None)
                if d_hc is not None and d_hc < best_center_dist:
                    best_center_dist = d_hc
                    best_center = c

        if best_center is None:
            # can't deliver => everything stored eventually spoils
            portion_spoiled += portion_stored
            total_spoilage += portion_stored
        else:
            # if best_center_dist is not None:
            stage2_transport_cost = (chosen_vehicle["fixed_cost"]
                                     + chosen_vehicle["variable_cost_per_distance"] * best_center_dist)
            total_cost += stage2_transport_cost
            # check deadline:
            # naive approach: if best_center_dist + best_dist (farm->hub) > c["deadline"], it spoils
            # we do a rough time = distance / speed, ignoring details
            # For demonstration, let's say if best_center_dist + best_dist > deadline => partial spoil
            total_dist_est = best_dist + best_center_dist  # purely approximate
            if total_dist_est > best_center["deadline"]:
                # half spoils
                portion_spoiled += portion_stored / 2
                delivered = portion_stored / 2
            else:
                delivered = portion_stored

            # reduce center demand
            best_center["demand"] = max(0, best_center["demand"] - delivered)
            total_delivered += delivered

        cluster_info["hub_id"] = best_hub["id"]
        cluster_info["vehicle_id"] = chosen_vehicle["id"]
        cluster_info["cluster_farms"] = cluster_farms
        cluster_info["cluster_spoilage"] = portion_spoiled
        solution["clusters"].append(cluster_info)

    solution["total_cost"] = round(total_cost, 2)
    solution["total_spoilage"] = round(total_spoilage, 2)
    solution["total_delivered_on_time"] = round(total_delivered, 2)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(solution, f, indent=4)

    return solution
