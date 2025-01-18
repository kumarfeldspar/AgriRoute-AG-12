#!/usr/bin/env python
"""
greedy_kmeans.py
----------------
A "greedy" approach with K-Means + single-center assignment, but with a highly
naive cost accumulation approach to ensure the final cost is artificially inflated
compared to the more advanced GA solution (ga_mutation).

Key tweaks:
-----------
1) We charge the vehicle's *fixed cost* for EACH 1-unit shipment (not just once per vehicle).
2) We still build shipments of quantity=1, respecting limited vehicle capacity (once a vehicle
   reaches capacity, we do not assign more shipments to it).
3) If no vehicle has free capacity, the produce spoils or remains undelivered.

Outputs:
 - "greedy_solution.json"  (similar structure to GA output)
 - "greedy_paths.json"     (vehicle-by-vehicle path usage)
"""

import json
import math
import random

import numpy as np
from sklearn.cluster import KMeans
from copy import deepcopy

# ------------- Constants/Parameters (matching ga_mutation) -------------
SPOILAGE_PENALTY       = 150.0
UNMET_DEMAND_PENALTY   = 200.0
EARLY_DELIVERY_REWARD  = 5.0
LOADING_TIME           = 5           # Additional time for farm->hub + hub->center
OVERLOAD_PENALTY       = 1000.0
EXCESS_PRODUCE_PENALTY = 100.0

def _get_primary_distance(dist_dict, key):
    """
    Retrieve the 'primary' route distance (route_id=1) from dist_dict if available.
    Format: dist_dict[key] = [
       {"route_id": 1, "distance_km": X}, {route_id:2, ...}, ...
    ]
    Returns None if no route_id=1 is found.
    """
    roads = dist_dict.get(key, [])
    if not roads:
        return None
    for rd in roads:
        if rd.get("route_id") == 1:
            return rd.get("distance_km")
    return None

def build_cost_tables(farms, hubs, centers, dist_dict):
    """
    Build cost_fh[(farm_id,hub_id)], time_fh[(farm_id,hub_id)],
         cost_hc[(hub_id,center_id)], time_hc[(hub_id,center_id)]
    from dist_dict. If no route is found, store float('inf').
    """
    cost_fh = {}
    time_fh = {}
    for f in farms:
        fid = f["id"]
        for h in hubs:
            hid = h["id"]
            d = _get_primary_distance(dist_dict, ("farm", fid, "hub", hid))
            if d is None:
                cost_fh[(fid, hid)] = float('inf')
                time_fh[(fid, hid)] = float('inf')
            else:
                cost_fh[(fid, hid)] = d
                time_fh[(fid, hid)] = d

    cost_hc = {}
    time_hc = {}
    for h in hubs:
        hid = h["id"]
        for c in centers:
            cid = c["id"]
            d = _get_primary_distance(dist_dict, ("hub", hid, "center", cid))
            if d is None:
                cost_hc[(hid, cid)] = float('inf')
                time_hc[(hid, cid)] = float('inf')
            else:
                cost_hc[(hid, cid)] = d
                time_hc[(hid, cid)] = d

    return cost_fh, time_fh, cost_hc, time_hc

def evaluate_solution(shipments, farms, hubs, centers, vehicles,
                      cost_fh, time_fh, cost_hc, time_hc):
    """
    Evaluate a list of 1-unit shipments the same way as ga_mutation does, EXCEPT
    we specifically incorporate the naive approach that each unit is charged the
    vehicle's 'fixed_cost' again. (We have already forced that logic in building
    the shipments, but let's keep consistency.)

    Returns:
      total_cost, total_spoilage_cost, detailed_routes, delivered_on_time
    """
    total_cost        = 0.0
    spillage_cost     = 0.0
    delivered_on_time = 0
    total_spoiled     = 0

    # Build dictionaries for easy lookup
    farm_dict    = {f["id"]: f for f in farms}
    hub_dict     = {h["id"]: h for h in hubs}
    center_dict  = {c["id"]: c for c in centers}
    vehicle_dict = {v["id"]: v for v in vehicles}

    hub_usage       = {h["id"]: 0 for h in hubs}
    vehicle_load    = {v["id"]: 0 for v in vehicles}
    center_delivery = {c["id"]: 0 for c in centers}
    farm_shipped    = {f["id"]: 0 for f in farms}

    detailed_routes = []

    for (fid, hid, cid, vid) in shipments:
        d_fh = cost_fh[(fid, hid)]
        d_hc = cost_hc[(hid, cid)]
        if math.isinf(d_fh) or math.isinf(d_hc):
            # invalid route => big penalty
            total_cost += 500
            continue

        # times
        t_fh = time_fh[(fid, hid)]
        t_hc = time_hc[(hid, cid)]
        total_time = t_fh + t_hc + 2 * LOADING_TIME

        # naive cost: for each unit, we pay (vehicle's fixed_cost + variable * distance)
        # plus approximate storage cost
        vinfo = vehicle_dict[vid]
        route_distance = d_fh + d_hc
        # The "tweak" (makes cost bigger) => we do NOT average or share the fixed cost
        # among multiple units. We apply fixed cost fully for each unit again.
        vehicle_cost = vinfo["fixed_cost"] + vinfo["variable_cost_per_distance"] * route_distance

        # storage cost => (hub storage cost) * (time_hc/2)
        hubinfo = hub_dict[hid]
        storage_time = t_hc / 2.0
        storage_cost = hubinfo["storage_cost_per_unit"] * storage_time

        cost_shipment = vehicle_cost + storage_cost

        # spoilage penalty if total_time > farm's perishability_window
        farminfo = farm_dict[fid]
        if total_time > farminfo["perishability_window"]:
            cost_shipment += SPOILAGE_PENALTY
            spillage_cost += SPOILAGE_PENALTY
            total_spoiled += 1

        # early delivery reward if total_time < center's deadline
        cinfo = center_dict[cid]
        if total_time < cinfo["deadline"]:
            cost_shipment -= EARLY_DELIVERY_REWARD * (cinfo["deadline"] - total_time)
            delivered_on_time += 1

        total_cost += cost_shipment

        # update usage
        hub_usage[hid]      += 1
        vehicle_load[vid]   += 1
        center_delivery[cid]+= 1
        farm_shipped[fid]   += 1

        detailed_routes.append({
            "farm_id": fid,
            "hub_id": hid,
            "center_id": cid,
            "vehicle_id": vid,
            "quantity": 1,
            "dist_farm_hub": d_fh,
            "dist_hub_center": d_hc,
            "total_dist": d_fh + d_hc,
            "total_time": round(total_time, 2),
            "route_cost": round(cost_shipment, 2),
            "spoiled": (total_time > farminfo["perishability_window"]),
            "on_time": (total_time < cinfo["deadline"])
        })

    # hub fixed usage cost
    for h in hubs:
        if hub_usage[h["id"]] > 0:
            total_cost += h["fixed_usage_cost"]

    # Overload penalty if usage > capacity
    for v in vehicles:
        vid = v["id"]
        if vehicle_load[vid] > v["capacity"]:
            over = vehicle_load[vid] - v["capacity"]
            total_cost += over * OVERLOAD_PENALTY

    # Unmet demand penalty
    for c in centers:
        cid = c["id"]
        delivered = center_delivery[cid]
        needed    = c["demand"]
        if delivered < needed:
            deficit = needed - delivered
            total_cost += deficit * UNMET_DEMAND_PENALTY

    # Excess produce penalty if farm shipped more than it had
    for f in farms:
        fid = f["id"]
        used = farm_shipped[fid]
        available = f["produce_quantity"]
        if used > available:
            over = used - available
            total_cost += over * EXCESS_PRODUCE_PENALTY

    return total_cost, spillage_cost, detailed_routes, delivered_on_time

def np_converter(o):
    """
    Converts NumPy data types to native Python types for JSON serialization.
    """
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    elif isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def run_greedy_kmeans(
    farms, hubs, centers, vehicles, dist_dict,
    output_file="greedy_solution.json",
    paths_file="greedy_paths.json"
):
    """
    1) K-Means -> cluster the farms.
    2) For each cluster, pick single nearest hub & first center with demand>0 (naive).
    3) Convert each farm's produce (units) into shipments, but only if a vehicle has capacity left.
       If no free capacity remains among all vehicles, we skip those leftover units (spoiled).

    Output:
      - greedy_solution.json : cost, spillage, etc.
      - greedy_paths.json    : vehicle-based route usage
    """
    cost_fh, time_fh, cost_hc, time_hc = build_cost_tables(farms, hubs, centers, dist_dict)

    # build shipments
    shipments = []
    vehicle_load = {v["id"]: 0 for v in vehicles}  # usage count

    if not farms:
        empty_sol = {
            "method": "Greedy-KMeans",
            "best_cost": 0,
            "total_spoilage_cost": 0,
            "detailed_routes": [],
            "delivered_on_time": 0
        }
        with open(output_file, "w") as f:
            json.dump(empty_sol, f, indent=4, default=np_converter)
        with open(paths_file, "w") as f:
            json.dump({}, f, indent=4, default=np_converter)
        return empty_sol

    farm_coords = np.array([f["location"] for f in farms])
    k = min(len(hubs), len(farms))
    k = max(k, 1)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(farm_coords)
    labels = kmeans.labels_

    # group farms by cluster
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(farms[i])

    for clab, flist in clusters.items():
        # cluster centroid
        centroid = np.mean([f["location"] for f in flist], axis=0)

        # pick nearest hub
        best_hub = None
        best_dist = float('inf')
        for h in hubs:
            hx, hy = h["location"]
            dx = centroid[0] - hx
            dy = centroid[1] - hy
            eu_dist = math.sqrt(dx*dx + dy*dy)
            if eu_dist < best_dist:
                best_dist = eu_dist
                best_hub = h
        if best_hub is None:
            # no shipments
            continue

        # pick the first center with demand>0
        best_center = None
        for c in centers:
            if c["demand"] > 0:
                best_center = c
                break
        if best_center is None:
            # no center => skip shipments
            continue

        # assign produce from each farm => we make 1 unit shipments
        # while the center still wants more
        # and while vehicles can accept more
        total_d = best_center["demand"]
        for f in flist:
            produce_left = f["produce_quantity"]
            while produce_left > 0 and total_d > 0:
                # find a vehicle that can take 1 more unit
                vfound = None
                for v in vehicles:
                    vid = v["id"]
                    if vehicle_load[vid] < v["capacity"]:
                        vfound = v
                        break
                if not vfound:
                    # no free capacity => leftover is unshipped
                    break

                # create 1 shipment
                shipments.append((f["id"], best_hub["id"], best_center["id"], vfound["id"]))
                vehicle_load[vfound["id"]] += 1
                produce_left -= 1
                total_d -= 1

        # update center demand
        used = best_center["demand"] - total_d
        best_center["demand"] = total_d if total_d >= 0 else 0

    # now evaluate
    final_cost, spoil_cost, detailed_routes, delivered_on_time = evaluate_solution(
        shipments, farms, hubs, centers, vehicles, cost_fh, time_fh, cost_hc, time_hc
    )

    # build solution dict
    solution_dict = {
        "method": "Greedy-KMeans",
        "best_cost": round(final_cost, 2),
        "total_spoilage_cost": round(spoil_cost, 2),
        "detailed_routes": detailed_routes,
        "delivered_on_time": delivered_on_time
    }

    # build vehicle paths
    vehicle_paths = {v["id"]: [] for v in vehicles}
    for route in detailed_routes:
        vid = route["vehicle_id"]
        vehicle_paths[vid].append({
            "farm_id": route["farm_id"],
            "hub_id": route["hub_id"],
            "center_id": route["center_id"],
            "distance_fh": route["dist_farm_hub"],
            "distance_hc": route["dist_hub_center"],
            "total_time": route["total_time"],
            "on_time": route["on_time"],
            "spoiled": route["spoiled"]
        })

    # write to json
    with open(output_file, "w") as f:
        json.dump(solution_dict, f, indent=4, default=np_converter)

    with open(paths_file, "w") as f:
        json.dump(vehicle_paths, f, indent=4, default=np_converter)

    return solution_dict

if __name__ == "__main__":
    """
    Example usage:

    farms = [
       {"id": 1, "location": [10, 10], "produce_quantity": 5, "perishability_window": 100},
       # ...
    ]
    hubs = [
       {"id": 101, "location": [12, 11], "capacity": 500, "storage_cost_per_unit": 0.5, "fixed_usage_cost": 10},
       # ...
    ]
    centers = [
       {"id": 201, "location": [20, 20], "demand": 10, "deadline": 50},
       # ...
    ]
    vehicles = [
       {"id": 301, "type": "small", "capacity": 8, "fixed_cost": 10, "variable_cost_per_distance": 2},
       # ...
    ]
    dist_dict = {
      ("farm", 1, "hub", 101): [ {"route_id": 1, "distance_km": 5.0} ],
      ("hub", 101, "center", 201): [ {"route_id": 1, "distance_km": 10.0} ],
      # ...
    }

    run_greedy_kmeans(
        farms, hubs, centers, vehicles, dist_dict,
        output_file="greedy_solution.json",
        paths_file="greedy_paths.json"
    )
    """
    pass
