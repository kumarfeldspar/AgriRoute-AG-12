#!/usr/bin/env python
"""
ga_mutation.py
--------------
Implements a Genetic Algorithm approach (using your "mutation algorithm") to solve
the same multi-stage farm->hub->center problem. Returns a solution dictionary with
detailed routes for each 1-unit shipment.
"""

import random
import math
import numpy as np
from copy import deepcopy
import json

# Penalties/Parameters (you can adjust these)
SPOILAGE_PENALTY = 150
UNMET_DEMAND_PENALTY = 200
EARLY_DELIVERY_REWARD = 5
LOADING_TIME = 5       # Additional time added farm->hub and hub->center
OVERLOAD_PENALTY = 1000  # heavy penalty per overloaded unit

POP_SIZE = 80
MAX_GENERATIONS = 100

def _pick_primary_distance(dist_list):
    """Pick route_id=1 from the list, or None if not found."""
    if not dist_list:
        return None
    for rd in dist_list:
        if rd["route_id"] == 1:
            return rd["distance_km"]
    return None

def build_cost_tables(farms, hubs, centers, dist_dict):
    """
    Construct cost_fh[(f,h)], time_fh[(f,h)],
             cost_hc[(h,c)], time_hc[(h,c)]
    from the dist_dict.
    If no route found, store float('inf').
    """
    cost_fh = {}
    time_fh = {}
    for f in farms:
        fid = f["id"]
        for h in hubs:
            hid = h["id"]
            d = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
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
            d = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
            if d is None:
                cost_hc[(hid, cid)] = float('inf')
                time_hc[(hid, cid)] = float('inf')
            else:
                cost_hc[(hid, cid)] = d
                time_hc[(hid, cid)] = d

    return cost_fh, time_fh, cost_hc, time_hc

# ============== GA Implementation ==============

def create_individual(num_shipments, farms, hubs, centers, vehicles):
    """
    Creates an individual solution as a list of shipments:
    each shipment is a tuple (farm_id, hub_id, center_id, vehicle_id).
    We'll pick random valid IDs for each gene.
    """
    all_farm_ids = [f["id"] for f in farms]
    all_hub_ids = [h["id"] for h in hubs]
    all_center_ids = [c["id"] for c in centers]
    all_vehicle_ids = [v["id"] for v in vehicles]

    individual = []
    for _ in range(num_shipments):
        f = random.choice(all_farm_ids)
        h = random.choice(all_hub_ids)
        c = random.choice(all_center_ids)
        v = random.choice(all_vehicle_ids)
        individual.append((f, h, c, v))
    return individual

def evaluate_individual(individual, farms, hubs, centers, vehicles, cost_fh, time_fh, cost_hc, time_hc):
    """
    Compute total cost for an individual solution.
    Each gene in `individual` is 1 unit from farm->hub->center using a specific vehicle.

    We'll accumulate:
      - Transport cost (vehicle fixed+variable).
      - Storage cost (some approximation).
      - Spoilage penalty if total_time > farm's perishability_window.
      - Unmet demand penalty if center deliveries < center demand.
      - Overload penalty if any vehicle is used above capacity.
      - Reward for early delivery if delivered time < center deadline.

    Returns the cost (lower is better).
    """
    total_cost = 0.0

    # Convert lists/dicts for easy lookup
    farm_dict = {f["id"]: f for f in farms}
    hub_dict = {h["id"]: h for h in hubs}
    center_dict = {c["id"]: c for c in centers}
    vehicle_dict = {v["id"]: v for v in vehicles}

    # Track usage
    hub_usage = {h["id"]: 0 for h in hubs}  # how many units pass
    vehicle_load = {v["id"]: 0 for v in vehicles}
    center_delivery = {c["id"]: 0 for c in centers}
    farm_shipment_count = {f["id"]: 0 for f in farms}  # how many units from each farm

    # Evaluate each 1-unit shipment
    for (fid, hid, cid, vid) in individual:
        # Distances
        d_fh = cost_fh[(fid, hid)]
        d_hc = cost_hc[(hid, cid)]
        # If infinite, it means no route => big penalty or treat as spoil
        if math.isinf(d_fh) or math.isinf(d_hc):
            # This shipment is effectively wasted
            total_cost += 500  # large penalty
            continue

        # Times
        t_fh = time_fh[(fid, hid)]
        t_hc = time_hc[(hid, cid)]
        total_time = t_fh + t_hc + 2 * LOADING_TIME

        # Basic transport cost:
        vinfo = vehicle_dict[vid]
        route_distance = d_fh + d_hc
        vehicle_cost = vinfo["fixed_cost"] + vinfo["variable_cost_per_distance"] * route_distance

        # Storage cost approximation: store at hub ~ half of hub->center time
        hubinfo = hub_dict[hid]
        storage_time = t_hc / 2.0
        storage_cost = hubinfo["storage_cost_per_unit"] * storage_time

        cost_shipment = vehicle_cost + storage_cost
        # Spoilage
        farminfo = farm_dict[fid]
        if total_time > farminfo["perishability_window"]:
            cost_shipment += SPOILAGE_PENALTY

        # Early Delivery Reward
        cinfo = center_dict[cid]
        if total_time < cinfo["deadline"]:
            # e.g. reward = EARLY_DELIVERY_REWARD * (deadline - total_time)
            cost_shipment -= EARLY_DELIVERY_REWARD * (cinfo["deadline"] - total_time)

        total_cost += cost_shipment

        # Update usage
        hub_usage[hid] += 1
        vehicle_load[vid] += 1
        center_delivery[cid] += 1
        farm_shipment_count[fid] += 1

    # Add fixed usage cost for any hub used
    for h in hubs:
        if hub_usage[h["id"]] > 0:
            total_cost += h["fixed_usage_cost"]

    # Check if farm shipments exceed produce
    for f in farms:
        fid = f["id"]
        if farm_shipment_count[fid] > f["produce_quantity"]:
            # penalty if we used more produce than available
            over = farm_shipment_count[fid] - f["produce_quantity"]
            total_cost += over * 100  # moderate penalty

    # Unmet demand penalty
    for c in centers:
        cid = c["id"]
        if center_delivery[cid] < c["demand"]:
            deficit = c["demand"] - center_delivery[cid]
            total_cost += deficit * UNMET_DEMAND_PENALTY

    # Overload penalty if vehicle load > capacity
    for v in vehicles:
        vid = v["id"]
        if vehicle_load[vid] > v["capacity"]:
            overload = vehicle_load[vid] - v["capacity"]
            total_cost += overload * OVERLOAD_PENALTY

    return total_cost

def fitness(individual, farms, hubs, centers, vehicles, cost_fh, time_fh, cost_hc, time_hc):
    """
    We convert cost to a fitness by 1/(cost+1e-6).
    """
    cost = evaluate_individual(individual, farms, hubs, centers, vehicles, cost_fh, time_fh, cost_hc, time_hc)
    return 1.0/(cost + 1e-6)

def roulette_wheel_selection(population, fitnesses):
    total = sum(fitnesses)
    pick = random.uniform(0, total)
    current = 0
    for ind, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return ind
    return population[-1]

def adaptive_probabilities(fit_value, avg_fit, best_fit, pc_range=(0.6, 0.9), pm_range=(0.001, 0.05)):
    """
    Compute adaptive crossover (pc) and mutation (pm) probabilities.
    """
    pc_min, pc_max = pc_range
    pm_min, pm_max = pm_range
    ratio = (fit_value - avg_fit)/(best_fit+1e-6)
    if fit_value >= avg_fit:
        # better than average => reduce mutation
        pc = pc_min + math.cos(ratio*(math.pi/2))*(pc_max - pc_min)
        pm = pm_min
    else:
        # below average => increase mutation
        pc = pc_max
        pm = pm_min + math.cos(ratio*(math.pi/2))*(pm_max - pm_min)
    return pc, pm

def crossover(p1, p2, pc):
    if random.random() > pc:
        return deepcopy(p1), deepcopy(p2)
    size = len(p1)
    if size < 2:
        return deepcopy(p1), deepcopy(p2)
    point1 = random.randint(0, size-2)
    point2 = random.randint(point1+1, size-1)
    c1 = p1[:point1] + p2[point1:point2] + p1[point2:]
    c2 = p2[:point1] + p1[point1:point2] + p2[point2:]
    return c1, c2

def mutate(ind, pm, farms, hubs, centers, vehicles):
    """
    With prob pm for each gene, replace it with a random shipment.
    """
    all_farm_ids = [f["id"] for f in farms]
    all_hub_ids = [h["id"] for h in hubs]
    all_center_ids = [c["id"] for c in centers]
    all_vehicle_ids = [v["id"] for v in vehicles]

    new_ind = deepcopy(ind)
    for i in range(len(new_ind)):
        if random.random() < pm:
            f = random.choice(all_farm_ids)
            h = random.choice(all_hub_ids)
            c = random.choice(all_center_ids)
            v = random.choice(all_vehicle_ids)
            new_ind[i] = (f, h, c, v)
    return new_ind

def run_genetic_algorithm(farms, hubs, centers, vehicles, dist_dict,
                          pop_size=POP_SIZE, max_generations=MAX_GENERATIONS,
                          output_file="ga_solution.json"):
    """
    Main entry point: run the improved GA, return a solution dictionary.
    """
    # 1) Build cost/time tables from dist_dict
    cost_fh, time_fh, cost_hc, time_hc = build_cost_tables(farms, hubs, centers, dist_dict)

    # 2) Number of shipments = sum of center demands
    total_required_shipments = sum(c["demand"] for c in centers)
    if total_required_shipments <= 0:
        # no shipments needed
        return {
            "method": "GeneticAlgorithm",
            "best_cost": 0,
            "detailed_routes": [],
            "total_spoilage": 0,
            "delivered_on_time": 0
        }

    # 3) Initialize population
    population = [create_individual(total_required_shipments, farms, hubs, centers, vehicles)
                  for _ in range(pop_size)]

    best_individual = None
    best_cost = float('inf')

    for g in range(max_generations):
        # Evaluate fitness
        fit_list = [fitness(ind, farms, hubs, centers, vehicles, cost_fh, time_fh, cost_hc, time_hc)
                    for ind in population]
        costs = [1.0/f - 1e-6 for f in fit_list]  # approximate invert
        avg_fit = sum(fit_list)/len(fit_list)
        best_idx = np.argmax(fit_list)
        current_best_cost = costs[best_idx]
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_individual = deepcopy(population[best_idx])

        # (Optional) print progress
        if g % 10 == 0:
            print(f"GA Generation {g} => Best cost so far: {best_cost:.2f}")

        current_best_fit = fit_list[best_idx]

        # Create next generation
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = roulette_wheel_selection(population, fit_list)
            p2 = roulette_wheel_selection(population, fit_list)
            pc, pm = adaptive_probabilities(fitness(p1, farms, hubs, centers, vehicles, cost_fh, time_fh, cost_hc, time_hc),
                                            avg_fit, current_best_fit)
            c1, c2 = crossover(p1, p2, pc)
            c1 = mutate(c1, pm, farms, hubs, centers, vehicles)
            c2 = mutate(c2, pm, farms, hubs, centers, vehicles)
            new_pop.extend([c1, c2])
        population = new_pop[:pop_size]

    # Evaluate final best
    final_cost = best_cost
    best_sol = best_individual

    # Build a detailed routes array
    # We'll reconstruct times, cost, on_time from best_sol
    # We can keep a running tally for spoilage and on-time deliveries:
    delivered_on_time = 0
    # For “spoilage,” we estimate how many shipments missed the window
    # but the cost function lumps spoilage penalty. We'll do a simpler measure: count how many shipments exceed window.
    total_spoil = 0

    farm_dict = {f["id"]: f for f in farms}
    hub_dict = {h["id"]: h for h in hubs}
    center_dict = {c["id"]: c for c in centers}
    vehicle_dict = {v["id"]: v for v in vehicles}

    detailed_routes = []
    for (fid, hid, cid, vid) in best_sol:
        d_fh = cost_fh[(fid, hid)]
        t_fh = time_fh[(fid, hid)]
        d_hc = cost_hc[(hid, cid)]
        t_hc = time_hc[(hid, cid)]
        total_time = t_fh + t_hc + 2*LOADING_TIME
        on_time = False
        if not math.isinf(d_fh) and not math.isinf(d_hc):
            if total_time <= center_dict[cid]["deadline"]:
                on_time = True
                delivered_on_time += 1
            if total_time > farm_dict[fid]["perishability_window"]:
                total_spoil += 1

        route_cost = 0  # optional if we want to list approximate cost
        # we won't recalc the entire cost here, just partial
        route_cost = (vehicle_dict[vid]["fixed_cost"] +
                      vehicle_dict[vid]["variable_cost_per_distance"] * (d_fh + d_hc) +
                      hub_dict[hid]["storage_cost_per_unit"]*(t_hc/2.0))

        detailed_routes.append({
            "farm_id": fid,
            "hub_id": hid,
            "center_id": cid,
            "vehicle_id": vid,
            "quantity": 1,
            "dist_farm_hub": d_fh if not math.isinf(d_fh) else None,
            "dist_hub_center": d_hc if not math.isinf(d_hc) else None,
            "total_dist": None if math.isinf(d_fh) or math.isinf(d_hc) else (d_fh + d_hc),
            "route_cost": round(route_cost, 3),
            "on_time": on_time
        })

    solution_dict = {
        "method": "GeneticAlgorithm",
        "best_cost": round(final_cost, 2),
        "detailed_routes": detailed_routes,
        "total_spoilage": total_spoil,           # number of shipments that spoiled
        "delivered_on_time": delivered_on_time   # number of shipments delivered on time
    }

    # Save to file
    with open(output_file, "w") as f:
        json.dump(solution_dict, f, indent=4)

    return solution_dict
