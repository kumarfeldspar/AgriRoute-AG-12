#!/usr/bin/env python
"""
greedy_kmeans.py
----------------
Simple Greedy + K-Means approach, ignoring advanced constraints.
Returns "greedy_solution.json" with route/cost/spoil data.
"""

import json
import math
import random

import numpy as np
from sklearn.cluster import KMeans

def _get_distance(dist_dict, key):
    """
    Retrieve the 'primary' route distance from dist_dict if available.
    dist_dict[key] = [ {route_id: 1, distance_km: X}, {route_id:2, ...}, ... ]
    We'll pick route_id=1 if it exists.
    """
    roads = dist_dict.get(key, [])
    if not roads:
        return None
    for rd in roads:
        if rd["route_id"] == 1:
            return rd["distance_km"]
    return None

def compute_storage_cost(units, hub_info, time_in_storage=1.0):
    """Storage cost: (units * cost_per_unit * time) + fixed_usage if used."""
    if units > 0:
        return units * hub_info["storage_cost_per_unit"] * time_in_storage + hub_info["fixed_usage_cost"]
    else:
        return 0.0

def run_greedy_kmeans(farms, hubs, centers, vehicles, dist_dict, output_file="greedy_solution.json"):
    """
    - K-Means cluster farms
    - Assign cluster total produce to nearest hub
    - Select smallest vehicle that can handle that load
    - Deliver to first center with demand > 0 (naive)
    - Output a dictionary with cost, spoilage, etc.
    """
    solution = {
        "method": "Greedy-KMeans",
        "clusters": [],
        "stage1_routes": [],
        "stage2_routes": [],
        "total_cost": 0.0,
        "total_spoilage": 0.0,
        "total_delivered_on_time": 0.0,
    }

    if not farms or not hubs or not centers:
        with open(output_file, "w") as f:
            json.dump(solution, f, indent=4)
        return solution

    # 1) K-means on farm coords
    farm_coords = np.array([f["location"] for f in farms])
    k = min(len(hubs), len(farms))
    if k < 1:
        k = 1
    kmeans = KMeans(n_clusters=k, random_state=42).fit(farm_coords)
    labels = kmeans.labels_

    # build clusters
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(farms[i])

    total_cost = 0.0
    total_spoilage = 0.0
    total_delivered = 0.0

    for clab, flist in clusters.items():
        cluster_info = {}
        cluster_farms = []
        total_cluster_produce = sum(f["produce_quantity"] for f in flist)
        centroid = np.mean([f["location"] for f in flist], axis=0)
        cluster_info["centroid"] = centroid.tolist()

        # find nearest hub to centroid (just Euclidian for approximate)
        best_hub = None
        best_dist = float("inf")
        for h in hubs:
            dist_euclid = math.dist(centroid, h["location"])
            if dist_euclid < best_dist:
                best_dist = dist_euclid
                best_hub = h

        if best_hub is None:
            # everything spoils
            for ff in flist:
                total_spoilage += ff["produce_quantity"]
                cluster_farms.append({
                    "farm_id": ff["id"],
                    "assigned_hub": None,
                    "quantity": ff["produce_quantity"],
                    "spoilage": ff["produce_quantity"]
                })
            cluster_info["hub_id"] = None
            cluster_info["vehicle_id"] = None
            cluster_info["cluster_spoilage"] = total_cluster_produce
            solution["clusters"].append(cluster_info)
            continue

        # check hub capacity
        if total_cluster_produce > best_hub["capacity"]:
            portion_stored = best_hub["capacity"]
            portion_spoiled = total_cluster_produce - portion_stored
        else:
            portion_stored = total_cluster_produce
            portion_spoiled = 0
        total_spoilage += portion_spoiled

        # pick a vehicle that can handle the portion_stored
        chosen_vehicle = None
        for v in sorted(vehicles, key=lambda x: x["capacity"]):
            if v["capacity"] >= portion_stored:
                chosen_vehicle = v
                break
        if chosen_vehicle is None:
            # all spoils
            portion_spoiled += portion_stored
            portion_stored = 0
            total_spoilage += portion_stored
            chosen_vehicle = {"id": None, "type": "none", "capacity": 0, "fixed_cost": 0, "variable_cost_per_distance": 0}

        # stage1 cost: sum over farms => farm->hub
        stage1_cost = 0.0
        for ff in flist:
            d_fh = _get_distance(dist_dict, ("farm", ff["id"], "hub", best_hub["id"]))
            if d_fh is None:
                # no route => spoils
                portion_spoiled += ff["produce_quantity"]
                total_spoilage += ff["produce_quantity"]
                assigned = 0
                sp = ff["produce_quantity"]
            else:
                assigned = ff["produce_quantity"]
                sp = 0
                # naive cost distribution
                cost_farm = (chosen_vehicle["fixed_cost"] + chosen_vehicle["variable_cost_per_distance"] * d_fh)
                cost_farm /= len(flist)  # distributing among cluster farms
                stage1_cost += cost_farm

            cluster_farms.append({
                "farm_id": ff["id"],
                "assigned_hub": best_hub["id"],
                "quantity": assigned,
                "spoilage": sp
            })
        total_cost += stage1_cost

        # storage cost
        st_cost = portion_stored * best_hub["storage_cost_per_unit"] + best_hub["fixed_usage_cost"] if portion_stored > 0 else 0
        total_cost += st_cost

        # stage2: deliver to the first center with demand > 0
        best_center = None
        best_center_dist = None
        for c in centers:
            if c["demand"] > 0:
                best_center = c
                d_hc = _get_distance(dist_dict, ("hub", best_hub["id"], "center", c["id"]))
                best_center_dist = d_hc
                break
        if best_center is None or best_center_dist is None:
            # spoils all that portion
            portion_spoiled += portion_stored
            total_spoilage += portion_stored
        else:
            stage2_cost = (chosen_vehicle["fixed_cost"] + chosen_vehicle["variable_cost_per_distance"] * best_center_dist)
            total_cost += stage2_cost
            # check deadline
            travel_approx = best_dist + best_center_dist
            if travel_approx > best_center["deadline"]:
                delivered = portion_stored/2
                portion_spoiled += portion_stored/2
            else:
                delivered = portion_stored
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

    with open(output_file, "w") as f:
        json.dump(solution, f, indent=4)

    return solution
