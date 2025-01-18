#!/usr/bin/env python
"""
ga_improved.py
--------------

Implements an "improved" Genetic Algorithm for the multi-stage farm->hub->center
distribution problem, using:
  - Roulette wheel selection
  - Adaptive crossover and mutation probabilities (based on cost)
  - A shipment-based representation (each gene = (farm_id, hub_id, center_id, vehicle_id))
  - **Elitism** to retain top solutions each generation

Outputs:
  1) ga_solution.json   (solution details)
  2) paths.json         (list of routes per vehicle)
"""

import random
import math
import numpy as np
from copy import deepcopy
import json

# ---------------- Global Penalties/Parameters ----------------
SPOILAGE_PENALTY      = 150
UNMET_DEMAND_PENALTY  = 200
EARLY_DELIVERY_REWARD = 5
LOADING_TIME          = 5       # Additional time from farm->hub, hub->center
OVERLOAD_PENALTY      = 1000    # Heavy penalty for each unit above vehicle capacity
EXCESS_PRODUCE_PENALTY= 100

# Genetic Algorithm parameters
POP_SIZE        = 80
MAX_GENERATIONS = 100

# Adaptive probability ranges (improved GA)
P_C1, P_C2 = 0.6, 0.9     # range for crossover probability
P_M1, P_M2 = 0.001, 0.05  # range for mutation probability


# ========================================================================
#                           Helper Functions
# ========================================================================
def _pick_primary_distance(dist_list):
    """
    Returns the distance for route_id=1 from dist_list, or None if not found.
    """
    if not dist_list:
        return None
    for rd in dist_list:
        if rd["route_id"] == 1:
            return rd["distance_km"]
    return None

def build_cost_tables(farms, hubs, centers, dist_dict):
    """
    Construct cost_fh[(f_id,h_id)], time_fh[(f_id,h_id)],
              cost_hc[(h_id,c_id)], time_hc[(h_id,c_id)]
    from the dist_dict.

    If no valid route is found for farm->hub or hub->center, store float('inf').
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

# ========================================================================
#                      GA Representation & Initialization
# ========================================================================
def create_individual(num_shipments, farms, hubs, centers, vehicles):
    """
    Creates a random individual as a list of shipments (f,h,c,v).
    Each shipment is 1 unit of produce from farm->hub->center with a chosen vehicle.
    """
    all_farm_ids    = [f["id"] for f in farms]
    all_hub_ids     = [h["id"] for h in hubs]
    all_center_ids  = [c["id"] for c in centers]
    all_vehicle_ids = [v["id"] for v in vehicles]

    individual = []
    for _ in range(int(num_shipments)):
        f = random.choice(all_farm_ids)
        h = random.choice(all_hub_ids)
        c = random.choice(all_center_ids)
        v = random.choice(all_vehicle_ids)
        individual.append((f, h, c, v))
    return individual

# ========================================================================
#                      Cost / Fitness Evaluation
# ========================================================================
def evaluate_individual(individual, farms, hubs, centers, vehicles,
                        cost_fh, time_fh, cost_hc, time_hc):
    """
    Compute the total cost of an individual solution, plus the spillage cost as a separate tally.

    The cost includes:
      - Vehicle transport (fixed + variable distance)
      - Storage cost at hub (approx)
      - Spoilage penalty if total_time > farm's perishability_window
      - Overload penalty (if more shipments assigned to a vehicle than capacity)
      - Unmet demand penalty (if center deliveries < demand)
      - Excess produce usage penalty (if farm shipments exceed produce_quantity)
      - Early delivery reward if total_time < center deadline
      - "No route" penalty if farm->hub or hub->center is inf

    Returns (total_cost, spillage_cost).
    """
    total_cost    = 0.0
    spillage_cost = 0.0

    farm_dict    = {f["id"]: f for f in farms}
    hub_dict     = {h["id"]: h for h in hubs}
    center_dict  = {c["id"]: c for c in centers}
    vehicle_dict = {v["id"]: v for v in vehicles}

    # usage trackers
    hub_usage       = {h["id"]: 0 for h in hubs}
    vehicle_load    = {v["id"]: 0 for v in vehicles}
    center_delivery = {c["id"]: 0 for c in centers}
    farm_shipped    = {f["id"]: 0 for f in farms}

    for (fid, hid, cid, vid) in individual:
        d_fh = cost_fh[(fid, hid)]
        d_hc = cost_hc[(hid, cid)]
        # no route => large penalty
        if math.isinf(d_fh) or math.isinf(d_hc):
            total_cost += 500
            continue

        # travel times
        t_fh = time_fh[(fid, hid)]
        t_hc = time_hc[(hid, cid)]
        total_time = t_fh + t_hc + 2 * LOADING_TIME

        # vehicle cost
        vinfo = vehicle_dict[vid]
        route_distance = d_fh + d_hc
        vehicle_cost   = vinfo["fixed_cost"] + (vinfo["variable_cost_per_distance"] * route_distance)

        # approximate storage cost => storing at hub half the t_hc
        hubinfo     = hub_dict[hid]
        storage_time= t_hc / 2.0
        storage_cost= hubinfo["storage_cost_per_unit"] * storage_time

        cost_shipment = vehicle_cost + storage_cost

        # spoil check
        farminfo = farm_dict[fid]
        if total_time > farminfo["perishability_window"]:
            cost_shipment += SPOILAGE_PENALTY
            spillage_cost += SPOILAGE_PENALTY  # track separately

        # early delivery reward
        cinfo = center_dict[cid]
        if total_time < cinfo["deadline"]:
            cost_shipment -= EARLY_DELIVERY_REWARD * (cinfo["deadline"] - total_time)

        total_cost += cost_shipment

        # usage update
        hub_usage[hid]        += 1
        vehicle_load[vid]     += 1
        center_delivery[cid]  += 1
        farm_shipped[fid]     += 1

    # add fixed usage cost for any used hub
    for h in hubs:
        if hub_usage[h["id"]] > 0:
            total_cost += h["fixed_usage_cost"]

    # overload penalty
    for v in vehicles:
        vid = v["id"]
        if vehicle_load[vid] > v["capacity"]:
            over = vehicle_load[vid] - v["capacity"]
            total_cost += over * OVERLOAD_PENALTY

    # unmet demand penalty
    for c in centers:
        cid = c["id"]
        needed    = c["demand"]
        delivered = center_delivery[cid]
        if delivered < needed:
            short = needed - delivered
            total_cost += short * UNMET_DEMAND_PENALTY

    # farm usage penalty (excess produce)
    for f in farms:
        fid = f["id"]
        assigned = farm_shipped[fid]
        if assigned > f["produce_quantity"]:
            over = assigned - f["produce_quantity"]
            total_cost += over * EXCESS_PRODUCE_PENALTY

    return total_cost, spillage_cost

def fitness_from_cost(cost):
    """
    Standard reciprocal fitness for a minimization problem => 1/(cost + 1e-6)
    """
    return 1.0 / (cost + 1e-6)

# ========================================================================
#                         Selection Operators
# ========================================================================
def roulette_wheel_selection(population, fitnesses):
    """
    Standard roulette-wheel selection.
    """
    total_fit = sum(fitnesses)
    pick      = random.uniform(0, total_fit)
    current   = 0.0
    for ind, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return ind
    return population[-1]

def compute_adaptive_probabilities(cost_i, cost_avg, cost_best,
                                   pc1=P_C1, pc2=P_C2,
                                   pm1=P_M1, pm2=P_M2):
    """
    Adaptive crossover & mutation probabilities based on individual's cost vs
    the average & best cost in the population.

    If cost_i < cost_avg => p_c = pc2, p_m = pm2
    else => use a cosine-based formula to reduce p_c or increase p_m
    """
    if cost_i < cost_avg:
        # better than average
        return pc2, pm2
    else:
        denom = abs(cost_best - cost_avg) + 1e-12
        ratio = (cost_i - cost_avg) / denom
        ratio = max(0.0, min(1.0, ratio))  # clamp to [0,1]

        p_c = pc1 + math.cos(ratio * (math.pi/2)) * (pc2 - pc1)
        p_m = pm1 + math.cos(ratio * (math.pi/2)) * (pm2 - pm1)
        return p_c, p_m

def crossover(p1, p2, p_c):
    """
    Standard 2-point crossover with probability p_c.
    """
    if random.random() > p_c:
        return deepcopy(p1), deepcopy(p2)

    size = len(p1)
    if size < 2:
        return deepcopy(p1), deepcopy(p2)

    c1, c2 = deepcopy(p1), deepcopy(p2)
    pt1 = random.randint(0, size - 2)
    pt2 = random.randint(pt1 + 1, size - 1)

    c1[pt1:pt2], c2[pt1:pt2] = p2[pt1:pt2], p1[pt1:pt2]
    return c1, c2

def mutate(ind, p_m, farms, hubs, centers, vehicles):
    """
    With probability p_m per gene, replace (f,h,c,v) with random valid combination.
    """
    all_farm_ids    = [f["id"] for f in farms]
    all_hub_ids     = [h["id"] for h in hubs]
    all_center_ids  = [c["id"] for c in centers]
    all_vehicle_ids = [v["id"] for v in vehicles]

    new_ind = deepcopy(ind)
    for i in range(len(new_ind)):
        if random.random() < p_m:
            f = random.choice(all_farm_ids)
            h = random.choice(all_hub_ids)
            c = random.choice(all_center_ids)
            v = random.choice(all_vehicle_ids)
            new_ind[i] = (f, h, c, v)
    return new_ind

# ========================================================================
#                         Main GA Loop (With Elitism)
# ========================================================================
def run_genetic_algorithm(farms, hubs, centers, vehicles, dist_dict,
                          pop_size=POP_SIZE,
                          max_generations=MAX_GENERATIONS,
                          output_file="ga_solution.json",
                          paths_file="paths.json"):
    """
    1) We compute #shipments = sum of center demands
    2) Build initial population of that many shipments each
    3) Evolve for max_generations
       - Evaluate cost => fitness
       - Elitism: carry the best TWO solutions to next generation
       - Fill the rest using roulette-wheel selection + adaptive cross/mutate
    4) Output best solution found
    """
    # build cost/time tables
    cost_fh, time_fh, cost_hc, time_hc = build_cost_tables(farms, hubs, centers, dist_dict)

    # how many shipments needed
    total_required_shipments = sum(c["demand"] for c in centers)
    if total_required_shipments <= 0:
        # trivial: no shipments
        empty_solution = {
            "method": "GeneticAlgorithm",
            "best_cost": 0,
            "total_spoilage_cost": 0,
            "detailed_routes": [],
            "delivered_on_time": 0
        }
        with open(output_file, "w") as f:
            json.dump(empty_solution, f, indent=4)
        with open(paths_file, "w") as f:
            json.dump({}, f, indent=4)
        return empty_solution

    # init population
    population = [
        create_individual(total_required_shipments, farms, hubs, centers, vehicles)
        for _ in range(pop_size)
    ]

    best_individual    = None
    best_cost          = float('inf')
    best_spoilage_cost = 0.0

    for gen in range(max_generations):
        # Evaluate cost & fitness
        costs = []
        spoil_costs = []
        for ind in population:
            c, sc = evaluate_individual(ind, farms, hubs, centers, vehicles,
                                        cost_fh, time_fh, cost_hc, time_hc)
            costs.append(c)
            spoil_costs.append(sc)

        fitnesses = [fitness_from_cost(c) for c in costs]

        # find best in this generation
        idx_best = np.argmin(costs)
        gen_best_cost = costs[idx_best]
        gen_best_ind  = population[idx_best]
        gen_best_spoil= spoil_costs[idx_best]

        if gen_best_cost < best_cost:
            best_cost          = gen_best_cost
            best_individual    = deepcopy(gen_best_ind)
            best_spoilage_cost = gen_best_spoil

        # (optional) console output
        if gen % 10 == 0:
            print(f"Gen {gen}: best_cost so far = {best_cost:.2f}")

        # -------------------------------------------------------------------
        #  Elitism: pick top TWO individuals from the old population
        # -------------------------------------------------------------------
        # sort population by cost ascending
        sorted_idx = sorted(range(len(population)), key=lambda i: costs[i])
        elite1_idx = sorted_idx[0]
        elite2_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]

        elite1 = deepcopy(population[elite1_idx])
        elite2 = deepcopy(population[elite2_idx])

        # -------------------------------------------------------------------
        #  Build next generation
        # -------------------------------------------------------------------
        new_population = [elite1, elite2]  # preserve the top 2 solutions

        while len(new_population) < pop_size:
            # pick parents by roulette
            p1 = roulette_wheel_selection(population, fitnesses)
            p2 = roulette_wheel_selection(population, fitnesses)

            # compute cost-based adaptive p_c, p_m for each parent, then average
            idx1 = population.index(p1)
            idx2 = population.index(p2)
            cost1= costs[idx1]
            cost2= costs[idx2]

            pc1, pm1 = compute_adaptive_probabilities(cost1, np.mean(costs), min(costs))
            pc2, pm2 = compute_adaptive_probabilities(cost2, np.mean(costs), min(costs))

            pc_use = (pc1 + pc2)/2.0
            pm_use = (pm1 + pm2)/2.0

            c1, c2 = crossover(p1, p2, pc_use)
            c1 = mutate(c1, pm_use, farms, hubs, centers, vehicles)
            c2 = mutate(c2, pm_use, farms, hubs, centers, vehicles)

            new_population.extend([c1, c2])

        population = new_population[:pop_size]

    # Evaluate best solution fully
    final_cost, final_spoilage_cost = evaluate_individual(
        best_individual, farms, hubs, centers, vehicles,
        cost_fh, time_fh, cost_hc, time_hc
    )

    # ---------------------------------------------------------
    #   Construct a "detailed_routes" array
    # ---------------------------------------------------------
    farm_dict    = {f["id"]: f for f in farms}
    hub_dict     = {h["id"]: h for h in hubs}
    center_dict  = {c["id"]: c for c in centers}
    vehicle_dict = {v["id"]: v for v in vehicles}

    delivered_on_time = 0
    total_spoil       = 0
    detailed_routes   = []

    # Also build the dictionary { vehicle_id: [ ... ] }
    vehicle_paths = {v["id"]: [] for v in vehicles}

    # Recompute times & cost for each shipment in best_individual
    for (fid, hid, cid, vid) in best_individual:
        d_fh = cost_fh[(fid, hid)]
        d_hc = cost_hc[(hid, cid)]
        t_fh = time_fh[(fid, hid)]
        t_hc = time_hc[(hid, cid)]
        total_time = float('inf')
        if (not math.isinf(d_fh)) and (not math.isinf(d_hc)):
            total_time = t_fh + t_hc + 2*LOADING_TIME

        # check spoil, on_time
        farminfo   = farm_dict[fid]
        centerinfo = center_dict[cid]

        is_spoiled = (total_time > farminfo["perishability_window"]) if total_time != float('inf') else True
        if is_spoiled:
            total_spoil += 1

        is_on_time = False
        if (total_time != float('inf')) and (total_time <= centerinfo["deadline"]):
            is_on_time = True
            delivered_on_time += 1

        # approximate route cost for reference
        route_cost = 0.0
        if (not math.isinf(d_fh)) and (not math.isinf(d_hc)):
            vinfo = vehicle_dict[vid]
            route_dist  = d_fh + d_hc
            vehicle_c   = (vinfo["fixed_cost"]
                          + vinfo["variable_cost_per_distance"] * route_dist)
            # hub storage cost ~ half of t_hc
            storage_c   = hub_dict[hid]["storage_cost_per_unit"]*(t_hc/2.0)
            route_cost  = vehicle_c + storage_c

        # record
        detailed_routes.append({
            "farm_id": fid,
            "hub_id": hid,
            "center_id": cid,
            "vehicle_id": vid,
            "quantity": 1,
            "dist_farm_hub": None if math.isinf(d_fh) else d_fh,
            "dist_hub_center": None if math.isinf(d_hc) else d_hc,
            "total_dist": None if (math.isinf(d_fh) or math.isinf(d_hc)) else (d_fh + d_hc),
            "total_time": None if (total_time == float('inf')) else round(total_time,2),
            "route_cost": round(route_cost,2),
            "on_time": is_on_time,
            "spoiled": is_spoiled
        })

        vehicle_paths[vid].append({
            "farm_id": fid,
            "hub_id": hid,
            "center_id": cid,
            "distance_fh": None if math.isinf(d_fh) else d_fh,
            "distance_hc": None if math.isinf(d_hc) else d_hc,
            "on_time": is_on_time,
            "spoiled": is_spoiled
        })

    solution_dict = {
        "method"             : "GeneticAlgorithm",
        "best_cost"          : round(final_cost, 2),
        "total_spoilage_cost": round(final_spoilage_cost, 2),
        "detailed_routes"    : detailed_routes,
        "delivered_on_time"  : delivered_on_time
    }

    # write solution
    with open(output_file, "w") as f:
        json.dump(solution_dict, f, indent=4)

    # write vehicle paths
    with open(paths_file, "w") as f:
        json.dump(vehicle_paths, f, indent=4)

    return solution_dict


if __name__ == "__main__":
    """
    Example usage:

    farms = [...]
    hubs = [...]
    centers = [...]
    vehicles = [...]
    dist_dict = {
       ("farm", f_id, "hub", h_id): [ {"route_id":1, "distance_km":...}, ...],
       ("hub", h_id, "center", c_id): [ {"route_id":1, "distance_km":...}, ...],
       ...
    }

    sol = run_genetic_algorithm(
        farms, hubs, centers, vehicles, dist_dict,
        pop_size=80, max_generations=100
    )
    print("Best GA solution cost:", sol["best_cost"])
    """
    pass
