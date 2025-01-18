#!/usr/bin/env python
"""
ga_improved.py
--------------

Implements an "improved" Genetic Algorithm for the multi-stage farm->hub->center
distribution problem, using:
  - Roulette wheel selection
  - Adaptive crossover and mutation probabilities (based on cost)
  - A shipment-based representation (each gene = (farm_id, hub_id, center_id, vehicle_id))

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
POP_SIZE         = 80
MAX_GENERATIONS  = 100

# Adaptive probability ranges (improved GA)
P_C1, P_C2 = 0.6, 0.9     # range for crossover probability
P_M1, P_M2 = 0.001, 0.05  # range for mutation probability

# -------------------------------------------------------------
def _pick_primary_distance(dist_list):
    """
    Returns the distance for route_id=1 from dist_list, or None if not found.
    This is just an example "primary route" picking function.
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

    If no valid route is found for farm->hub or hub->center, we store float('inf').
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

# -------------------------------------------------------------
#           GA Representation and Initialization
# -------------------------------------------------------------
def create_individual(num_shipments, farms, hubs, centers, vehicles):
    """
    Creates a random individual as a list of shipments:
    each shipment is a tuple (farm_id, hub_id, center_id, vehicle_id).
    """
    all_farm_ids   = [f["id"] for f in farms]
    all_hub_ids    = [h["id"] for h in hubs]
    all_center_ids = [c["id"] for c in centers]
    all_vehicle_ids= [v["id"] for v in vehicles]

    individual = []
    for _ in range((int(num_shipments))):
        f = random.choice(all_farm_ids)
        h = random.choice(all_hub_ids)
        c = random.choice(all_center_ids)
        v = random.choice(all_vehicle_ids)
        individual.append((f, h, c, v))
    return individual

# -------------------------------------------------------------
#           Cost / Fitness Evaluation
# -------------------------------------------------------------
def evaluate_individual(individual, farms, hubs, centers, vehicles,
                        cost_fh, time_fh, cost_hc, time_hc):
    """
    Compute the total cost of an individual solution.
    Also returns "spoilage_cost" as a separate tally (helpful if you want it reported).

    We accumulate:
      - Vehicle transport cost (fixed + variable distance).
      - Storage cost at the hub (approx).
      - Spoilage penalty if total_time > farm's perishability_window.
      - Unmet demand penalty if center deliveries < demand (handled outside shipments loop).
      - Overload penalty if a vehicle carries more units than capacity.
      - Excess produce usage penalty if a farm is used more than available quantity.
      - Reward for early delivery if total_time < center's deadline.

    Returns:
      total_cost, spillage_cost
    """
    total_cost       = 0.0
    spillage_cost    = 0.0

    farm_dict    = {f["id"]: f for f in farms}
    hub_dict     = {h["id"]: h for h in hubs}
    center_dict  = {c["id"]: c for c in centers}
    vehicle_dict = {v["id"]: v for v in vehicles}

    hub_usage      = {h["id"]: 0 for h in hubs}
    vehicle_load   = {v["id"]: 0 for v in vehicles}
    center_delivery= {c["id"]: 0 for c in centers}
    farm_shipped   = {f["id"]: 0 for f in farms}  # how many units shipped from each farm

    # Evaluate each shipment (1 unit)
    for (fid, hid, cid, vid) in individual:
        d_fh = cost_fh[(fid, hid)]
        d_hc = cost_hc[(hid, cid)]
        # If either segment is infinite => no route => penalize heavily
        if math.isinf(d_fh) or math.isinf(d_hc):
            total_cost += 500  # large penalty for invalid route
            continue

        # Travel times
        t_fh = time_fh[(fid, hid)]
        t_hc = time_hc[(hid, cid)]
        total_time = t_fh + t_hc + 2 * LOADING_TIME  # Add loading time at farm->hub and hub->center

        # Vehicle cost
        vinfo = vehicle_dict[vid]
        route_distance = d_fh + d_hc
        vehicle_cost   = (vinfo["fixed_cost"] +
                          vinfo["variable_cost_per_distance"] * route_distance)

        # Storage cost (approx) => store cost at hub ~ half of the time from hub->center
        hubinfo     = hub_dict[hid]
        storage_time= t_hc / 2.0
        storage_cost= hubinfo["storage_cost_per_unit"] * storage_time

        cost_shipment = vehicle_cost + storage_cost

        # Spoilage penalty
        farminfo = farm_dict[fid]
        if total_time > farminfo["perishability_window"]:
            cost_shipment   += SPOILAGE_PENALTY
            spillage_cost   += SPOILAGE_PENALTY  # track separately

        # Early Delivery Reward
        cinfo = center_dict[cid]
        if total_time < cinfo["deadline"]:
            cost_shipment -= EARLY_DELIVERY_REWARD * (cinfo["deadline"] - total_time)

        total_cost += cost_shipment

        # Update usage tallies
        hub_usage[hid]        += 1
        vehicle_load[vid]     += 1
        center_delivery[cid]  += 1
        farm_shipped[fid]     += 1

    # Fixed usage cost for any hub used
    for h in hubs:
        if hub_usage[h["id"]] > 0:
            total_cost += h["fixed_usage_cost"]

    # Overload penalty if vehicle usage > capacity
    for v in vehicles:
        vid = v["id"]
        load = vehicle_load[vid]
        cap  = v["capacity"]
        if load > cap:
            over = load - cap
            total_cost += over * OVERLOAD_PENALTY

    # Unmet demand penalty
    for c in centers:
        cid = c["id"]
        delivered = center_delivery[cid]
        needed    = c["demand"]
        if delivered < needed:
            deficit = needed - delivered
            total_cost += deficit * UNMET_DEMAND_PENALTY

    # Excess produce usage
    for f in farms:
        fid = f["id"]
        used = farm_shipped[fid]
        available = f["produce_quantity"]
        if used > available:
            over = used - available
            total_cost += over * EXCESS_PRODUCE_PENALTY

    return total_cost, spillage_cost

def fitness_from_cost(cost):
    """
    Standard reciprocal fitness for a minimization problem:
       fitness = 1/(cost + 1e-6).
    """
    return 1.0/(cost + 1e-6)

# -------------------------------------------------------------
#             Roulette Wheel Selection
# -------------------------------------------------------------
def roulette_wheel_selection(population, fitnesses):
    """
    Standard roulette-wheel (proportional) selection.
    """
    total_fit = sum(fitnesses)
    pick      = random.uniform(0, total_fit)
    current   = 0
    for ind, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return ind
    return population[-1]  # fallback

# -------------------------------------------------------------
#     Adaptive Crossover & Mutation (Improved GA)
# -------------------------------------------------------------
def compute_adaptive_probabilities(cost_i, cost_avg, cost_best,
                                   pc1=P_C1, pc2=P_C2,
                                   pm1=P_M1, pm2=P_M2):
    """
    Compute the adaptive crossover (p_c) and mutation (p_m) probabilities
    based on the individual's cost vs. the average and the best (minimum) cost.

    In the improved approach:
      - Let f'_i = cost_i
      - Let f'_avg = cost_avg
      - Let f'_min = cost_best  (the smallest cost in the population)

      If cost_i < cost_avg  (better than avg):
          p_c = pc2
          p_m = pm2
      Else (worse than or equal to avg):
          ratio = (cost_i - cost_avg) / ((cost_best - cost_avg) + 1e-12)
          # NB: cost_best < cost_avg, so (cost_best - cost_avg) is negative.
          # We'll take the absolute value to keep ratio in [0,1].
          ratio = (cost_i - cost_avg)/abs(cost_best - cost_avg)

          # Then use the cosine function
          p_c = pc1 + math.cos(ratio * (math.pi/2))*(pc2 - pc1)
          p_m = pm1 + math.cos(ratio * (math.pi/2))*(pm2 - pm1)

    Adjust these formulas as needed to match your exact paper’s definitions
    (some references invert the sign or reorder cost_best/cost_avg).
    """
    # We assume cost_best <= cost_avg typically (the best cost is smaller).
    if cost_i < cost_avg:
        # better than average => intensify exploitation
        return pc2, pm2
    else:
        # worse than average => adapt with cos
        denom = abs(cost_best - cost_avg) + 1e-12
        ratio = (cost_i - cost_avg) / denom
        # clamp ratio to [0, 1] just in case
        ratio = max(0.0, min(1.0, ratio))

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

    # exchange segments p1[pt1:pt2], p2[pt1:pt2]
    c1[pt1:pt2], c2[pt1:pt2] = p2[pt1:pt2], p1[pt1:pt2]
    return c1, c2

def mutate(ind, p_m, farms, hubs, centers, vehicles):
    """
    Mutation: with probability p_m for each gene,
    replace (farm, hub, center, vehicle) with a random valid combination.
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

# -------------------------------------------------------------
#                  Main GA Loop
# -------------------------------------------------------------
def run_genetic_algorithm(farms, hubs, centers, vehicles, dist_dict,
                          pop_size=POP_SIZE,
                          max_generations=MAX_GENERATIONS,
                          output_file="ga_solution.json",
                          paths_file="paths.json"):
    """
    Main entry point for the improved GA.

    Returns a dictionary with:
      {
        "method": "GeneticAlgorithm",
        "best_cost": ...,
        "total_spoilage_cost": ...,
        "detailed_routes": [...],
        "delivered_on_time": ...
      }
    and also writes that dict to ga_solution.json (by default).
    Additionally writes the path of all vehicles in paths.json.
    """

    # 1) Build cost/time tables
    cost_fh, time_fh, cost_hc, time_hc = build_cost_tables(farms, hubs, centers, dist_dict)

    # 2) Number of shipments = sum of center demands
    total_required_shipments = sum(c["demand"] for c in centers)
    if total_required_shipments <= 0:
        # no shipments needed
        empty_solution = {
            "method": "GeneticAlgorithm",
            "best_cost": 0,
            "total_spoilage_cost": 0,
            "detailed_routes": [],
            "delivered_on_time": 0
        }
        with open(output_file, "w") as f:
            json.dump(empty_solution, f, indent=4)
        # paths.json can be empty
        with open(paths_file, "w") as f:
            json.dump({}, f, indent=4)
        return empty_solution

    # 3) Initialize population
    population = [
        create_individual(total_required_shipments, farms, hubs, centers, vehicles)
        for _ in range(pop_size)
    ]

    best_individual      = None
    best_cost            = float('inf')
    best_spoilage_cost   = 0.0

    for gen in range(max_generations):
        # --- Evaluate costs & fitness for each individual ---
        costs = []
        spoil_costs = []
        for ind in population:
            c, sc = evaluate_individual(ind, farms, hubs, centers, vehicles,
                                        cost_fh, time_fh, cost_hc, time_hc)
            costs.append(c)
            spoil_costs.append(sc)

        # fitness = 1/(cost+1e-6)
        fitnesses = [fitness_from_cost(c) for c in costs]

        # track best
        min_cost      = min(costs)
        avg_cost      = sum(costs)/len(costs)
        idx_best      = costs.index(min_cost)
        best_of_gen   = population[idx_best]
        best_spoilage = spoil_costs[idx_best]

        if min_cost < best_cost:
            best_cost          = min_cost
            best_individual    = deepcopy(best_of_gen)
            best_spoilage_cost = best_spoilage

        # (optional) console progress
        if gen % 10 == 0:
            print(f"Gen {gen} => best_cost so far: {best_cost:.2f}")

        # --- Roulette Wheel Selection (to build new population) ---
        new_population = []
        while len(new_population) < pop_size:
            p1 = roulette_wheel_selection(population, fitnesses)
            p2 = roulette_wheel_selection(population, fitnesses)

            # We also want to adapt p_c, p_m individually for each parent,
            # based on their *cost* (not their combined cost).
            # We'll pick cost of p1 to define p_c1, p_m1, cost of p2 => p_c2, p_m2,
            # then unify them in crossover/mutation steps.
            idx1 = population.index(p1)
            idx2 = population.index(p2)
            cost1= costs[idx1]
            cost2= costs[idx2]

            # compute adaptive probabilities for each parent
            pc1, pm1 = compute_adaptive_probabilities(cost1, avg_cost, min_cost)
            pc2, pm2 = compute_adaptive_probabilities(cost2, avg_cost, min_cost)

            # For simplicity, let us average the parents' pc, pm for the crossover step:
            pc_use = (pc1 + pc2)/2.0
            pm_use = (pm1 + pm2)/2.0

            # --- Crossover & Mutation ---
            c1, c2 = crossover(p1, p2, pc_use)
            c1 = mutate(c1, pm_use, farms, hubs, centers, vehicles)
            c2 = mutate(c2, pm_use, farms, hubs, centers, vehicles)

            new_population.extend([c1, c2])

        population = new_population[:pop_size]

    # Evaluate best solution found
    final_cost, final_spoilage_cost = evaluate_individual(
        best_individual, farms, hubs, centers, vehicles,
        cost_fh, time_fh, cost_hc, time_hc
    )

    # ---------------------------------------------------------
    #   Construct a "detailed_routes" array for the best solution
    # ---------------------------------------------------------
    farm_dict    = {f["id"]: f for f in farms}
    hub_dict     = {h["id"]: h for h in hubs}
    center_dict  = {c["id"]: c for c in centers}
    vehicle_dict = {v["id"]: v for v in vehicles}

    delivered_on_time = 0
    total_spoil       = 0
    detailed_routes   = []

    # We'll also build a dictionary { vehicle_id: [ {shipment details}, ... ] }
    vehicle_paths = { v["id"]: [] for v in vehicles }

    for (fid, hid, cid, vid) in best_individual:
        d_fh = cost_fh[(fid, hid)]
        t_fh = time_fh[(fid, hid)]
        d_hc = cost_hc[(hid, cid)]
        t_hc = time_hc[(hid, cid)]
        total_time = float('inf')
        if (not math.isinf(d_fh)) and (not math.isinf(d_hc)):
            total_time = t_fh + t_hc + 2*LOADING_TIME

        farminfo   = farm_dict[fid]
        centerinfo = center_dict[cid]

        # Did it spoil?
        spoiled = (total_time > farminfo["perishability_window"]) if (total_time != float('inf')) else True
        if spoiled:
            total_spoil += 1

        # On-time?
        on_time = False
        if total_time != float('inf') and (total_time <= centerinfo["deadline"]):
            on_time = True
            delivered_on_time += 1

        # approximate cost for each route (not strictly needed, but helpful):
        route_cost = 0.0
        if (not math.isinf(d_fh)) and (not math.isinf(d_hc)):
            vinfo = vehicle_dict[vid]
            route_dist  = d_fh + d_hc
            vehicle_c   = vinfo["fixed_cost"] + vinfo["variable_cost_per_distance"]*route_dist
            storage_c   = hub_dict[hid]["storage_cost_per_unit"]*(t_hc/2.0)
            route_cost  = vehicle_c + storage_c

        # Add to "detailed_routes"
        detailed_routes.append({
            "farm_id"         : fid,
            "hub_id"          : hid,
            "center_id"       : cid,
            "vehicle_id"      : vid,
            "quantity"        : 1,
            "dist_farm_hub"   : None if math.isinf(d_fh) else d_fh,
            "dist_hub_center" : None if math.isinf(d_hc) else d_hc,
            "total_dist"      : None if (math.isinf(d_fh) or math.isinf(d_hc)) else (d_fh + d_hc),
            "total_time"      : None if (total_time == float('inf')) else round(total_time,2),
            "route_cost"      : round(route_cost,2),
            "on_time"         : on_time,
            "spoiled"         : spoiled
        })

        # Also record in vehicle_paths
        vehicle_paths[vid].append({
            "farm_id": fid,
            "hub_id" : hid,
            "center_id": cid,
            "distance_fh": None if math.isinf(d_fh) else d_fh,
            "distance_hc": None if math.isinf(d_hc) else d_hc,
            "on_time": on_time,
            "spoiled": spoiled
        })

    solution_dict = {
        "method"               : "GeneticAlgorithm",
        "best_cost"            : round(final_cost, 2),
        "total_spoilage_cost"  : round(final_spoilage_cost, 2),
        "detailed_routes"      : detailed_routes,
        "delivered_on_time"    : delivered_on_time
    }

    # write solution JSON
    with open(output_file, "w") as f:
        json.dump(solution_dict, f, indent=4)

    # write vehicle paths to paths.json
    with open(paths_file, "w") as f:
        json.dump(vehicle_paths, f, indent=4)

    return solution_dict


# ------------------------- Usage Example -------------------------
if __name__ == "__main__":
    # Suppose you already have your farms, hubs, centers, vehicles, and dist_dict loaded
    # from somewhere (JSON, database, etc.). You'd do something like:
    #
    # farms = [...]
    # hubs = [...]
    # centers = [...]
    # vehicles = [...]
    # dist_dict = {("farm", f_id, "hub", h_id): [{...}, {...}], ...} etc.
    #
    # Then just call:
    #
    # sol = run_genetic_algorithm(
    #     farms, hubs, centers, vehicles, dist_dict,
    #     pop_size=80, max_generations=100,
    #     output_file="ga_solution.json", paths_file="paths.json"
    # )
    #
    # print("Best GA solution cost:", sol["best_cost"])
    pass
