#!/usr/bin/env python
"""
linear_programming.py
---------------------
LP-based approach with SPOILAGE_PENALTY, UNMET_DEMAND_PENALTY, etc.

Generates a JSON with the structure similar to GA_Mutation output,
one line per unit (or partial unit) shipped:

{
  "method": "LinearProgramming",
  "best_cost": <float>,
  "total_spoilage_cost": <float>,
  "detailed_routes": [
    {
       "farm_id": <int>,
       "hub_id": <int>,
       "center_id": <int>,
       "vehicle_id": <int>,
       "quantity": <float>,  # often 1.0 or a fraction
       "dist_farm_hub": <float>,
       "dist_hub_center": <float>,
       "total_dist": <float>,
       "total_time": <float>,
       "route_cost": <float>,
       "on_time": <bool>,
       "spoiled": <bool>
    },
    ...
  ]
}
"""

import json
import math
import time

# Constants
SPOILAGE_PENALTY      = 150
UNMET_DEMAND_PENALTY  = 200
EARLY_DELIVERY_REWARD = 5
LOADING_TIME          = 5
OVERLOAD_PENALTY      = 1000
EXCESS_PRODUCE_PENALTY= 100

try:
    from docplex.mp.model import Model
    HAVE_DOCPLEX = True
except ImportError:
    HAVE_DOCPLEX = False
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpContinuous, LpStatus, value

BIG_M = 999999

def _pick_primary_distance(dist_list):
    """
    Return the distance_km for route_id=1 if found, else None.
    """
    if not dist_list:
        return None
    for rd in dist_list:
        if rd.get("route_id", None) == 1:
            return rd.get("distance_km", None)
    return None

def run_linear_program(
    farms, hubs, centers, vehicles, dist_dict, output_file="lp_solution.json"
):
    """
    Build & solve the LP, returning a dict with:
    {
      "method": "LinearProgramming",
      "best_cost": <float>,
      "total_spoilage_cost": <float>,
      "detailed_routes": [...]
    }

    It expands each route variable's quantity into single-unit shipments (or
    partial leftover) to match the style of your GA solutions.
    """
    start_time = time.time()

    if not HAVE_DOCPLEX:
        raise RuntimeError(
            "docplex is not installed. Please install it or adapt code for PuLP fallback."
        )

    solution = _run_docplex_lp(farms, hubs, centers, vehicles, dist_dict)
    solution["runtime_sec"] = round(time.time() - start_time, 2)

    # Save solution to JSON
    with open(output_file, "w") as f:
        json.dump(solution, f, indent=4)

    return solution


def _run_docplex_lp(farms, hubs, centers, vehicles, dist_dict):
    """
    1) Variables:
        x[(f,h,c,v)] = quantity delivered farm->hub->center with vehicle v
        spoil_f[f]   = quantity spoiled at farm f
        unmet_c[c]   = unmet demand at center c

    2) Constraints:
        - Vehicle capacity (sum of x <= capacity)
        - Hub capacity
        - Farm production = sum of x + spoil
        - Center demand = sum of x + unmet
        - If (dist_total + 2*LOADING_TIME) > farm's window => x=0

    3) Objective:
       Minimize [Transport + Storage + (SPOILAGE_PENALTY + EXCESS_PRODUCE_PENALTY)*spoil_f + UNMET_DEMAND_PENALTY*unmet_c]
       minus [EARLY_DELIVERY_REWARD * x if delivered in time].
    """
    # Create model
    m = Model(name="AgroRouteLP_docplex")

    # Decision variables
    x = {}            # route shipments
    spoil_f = {}      # spoilage
    unmet_c = {}      # unmet demand

    # Build variables
    for f in farms:
        fid = f["id"]
        spoil_f[fid] = m.continuous_var(lb=0, name=f"spoil_{fid}")

    for c in centers:
        cid = c["id"]
        unmet_c[cid] = m.continuous_var(lb=0, name=f"unmet_{cid}")

    for f in farms:
        fid = f["id"]
        for h in hubs:
            hid = h["id"]
            dist_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
            if dist_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                dist_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
                if dist_hc is None:
                    continue
                for v in vehicles:
                    vid = v["id"]
                    x[(fid, hid, cid, vid)] = m.continuous_var(
                        lb=0, name=f"x_{fid}_{hid}_{cid}_{vid}"
                    )

    # Constraints
    # a) Vehicle capacity
    for v in vehicles:
        vid = v["id"]
        cap = v["capacity"]
        expr = m.sum(x[k] for k in x if k[3] == vid)
        m.add_constraint(expr <= cap, ctname=f"veh_cap_{vid}")

    # b) Hub capacity
    for h in hubs:
        hid = h["id"]
        cap = h["capacity"]
        expr = m.sum(x[k] for k in x if k[1] == hid)
        m.add_constraint(expr <= cap, ctname=f"hub_cap_{hid}")

    # c) Farm production
    for f in farms:
        fid = f["id"]
        qty = f["produce_quantity"]
        expr = m.sum(x[k] for k in x if k[0] == fid)
        m.add_constraint(expr + spoil_f[fid] == qty, ctname=f"farm_qty_{fid}")

    # d) Center demand
    for c in centers:
        cid = c["id"]
        dem = c["demand"]
        expr = m.sum(x[k] for k in x if k[2] == cid)
        m.add_constraint(expr + unmet_c[cid] == dem, ctname=f"center_demand_{cid}")

    # e) Perishability window
    # If total_time > farm's window => force x=0
    for f in farms:
        fid = f["id"]
        pw = f["perishability_window"]
        for h in hubs:
            hid = h["id"]
            dist_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
            if dist_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                dist_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
                if dist_hc is None:
                    continue
                total_time = dist_fh + dist_hc + 2 * LOADING_TIME
                if total_time > pw:
                    # Then x=0 for all vehicles
                    for v in vehicles:
                        vid = v["id"]
                        if (fid, hid, cid, vid) in x:
                            m.add_constraint(x[(fid, hid, cid, vid)] == 0)

    # Build objective
    transport_terms = []
    storage_terms = []
    spoil_terms = []
    unmet_terms = []
    reward_terms = []

    # function for storage cost
    def storage_cost_pu(hub):
        # e.g. spread hub fixed_usage_cost over capacity
        ccap = max(1, hub["capacity"])
        return hub["storage_cost_per_unit"] + (hub["fixed_usage_cost"] / ccap)

    # spoil cost
    for f in farms:
        fid = f["id"]
        # SPOILAGE_PENALTY + EXCESS_PRODUCE_PENALTY
        spoil_terms.append((SPOILAGE_PENALTY + EXCESS_PRODUCE_PENALTY)*spoil_f[fid])

    # unmet cost
    for c in centers:
        cid = c["id"]
        unmet_terms.append(UNMET_DEMAND_PENALTY * unmet_c[cid])

    # transport + storage + reward
    for (fid, hid, cid, vid), var in x.items():
        dist_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
        dist_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
        dist_total = dist_fh + dist_hc

        # transport cost per unit
        vobj = next(V for V in vehicles if V["id"] == vid)
        cost_pu = (vobj["fixed_cost"] + dist_total*vobj["variable_cost_per_distance"]) / max(1, vobj["capacity"])
        transport_terms.append(cost_pu * var)

        # storage cost
        hobj = next(H for H in hubs if H["id"] == hid)
        scost_pu = storage_cost_pu(hobj)
        storage_terms.append(scost_pu * var)

        # check if on-time => reward
        cobj = next(C for C in centers if C["id"] == cid)
        total_time = dist_total + 2*LOADING_TIME
        if total_time <= cobj["deadline"]:
            reward_terms.append(EARLY_DELIVERY_REWARD * var)

    # sum them
    transport_cost_expr = m.sum(transport_terms)
    storage_cost_expr   = m.sum(storage_terms)
    spoil_cost_expr     = m.sum(spoil_terms)
    unmet_cost_expr     = m.sum(unmet_terms)
    reward_expr         = m.sum(reward_terms)

    total_cost_expr = transport_cost_expr + storage_cost_expr + spoil_cost_expr + unmet_cost_expr
    objective_expr  = total_cost_expr - reward_expr

    m.minimize(objective_expr)

    # Solve
    sol = m.solve(log_output=False)
    if sol is None:
        # No feasible solution
        return {
            "method": "LinearProgramming",
            "best_cost": None,
            "total_spoilage_cost": None,
            "detailed_routes": []
        }

    # Evaluate objective
    best_obj_val = objective_expr.solution_value
    spoil_val    = spoil_cost_expr.solution_value  # portion from spoilage

    # We'll define "best_cost" = the final objective
    best_cost = best_obj_val
    # "total_spoilage_cost" = portion from spoilage
    total_spoilage_cost = spoil_val

    # ----------------------------------------------------
    # Build expanded "detailed_routes" for each x>0 route
    # so that each unit (or leftover fraction) is a record
    # ----------------------------------------------------
    detailed_routes = []

    def compute_route_line(fid, hid, cid, vid, q):
        """
        Compute cost/time fields for 'q' quantity from (f->h->c) via v.
        Returns a dictionary for a single shipment (which might be 1.0 or fraction).
        """
        dist_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
        dist_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
        dist_total = dist_fh + dist_hc
        total_time = dist_total + 2*LOADING_TIME

        # Cost per unit
        vobj = next(V for V in vehicles if V["id"] == vid)
        cost_pu = (vobj["fixed_cost"] + dist_total*vobj["variable_cost_per_distance"]) / max(1, vobj["capacity"])

        hobj = next(H for H in hubs if H["id"] == hid)
        scost_pu = storage_cost_pu(hobj)

        base_cost_per_unit = cost_pu + scost_pu

        # early reward
        cobj = next(C for C in centers if C["id"] == cid)
        on_time = (total_time <= cobj["deadline"])
        # If on_time => subtract EARLY_DELIVERY_REWARD from cost
        final_cost_per_unit = base_cost_per_unit
        if on_time:
            final_cost_per_unit -= EARLY_DELIVERY_REWARD

        # check "spoiled"
        fobj = next(F for F in farms if F["id"] == fid)
        spoiled = (total_time > fobj["perishability_window"])

        # route cost
        route_cost = final_cost_per_unit * q

        return {
            "farm_id": fid,
            "hub_id": hid,
            "center_id": cid,
            "vehicle_id": vid,
            "quantity": round(q, 3),
            "dist_farm_hub": round(dist_fh, 3),
            "dist_hub_center": round(dist_hc, 3),
            "total_dist": round(dist_total, 3),
            "total_time": round(total_time, 3),
            "route_cost": round(route_cost, 2),
            "on_time": on_time,
            "spoiled": spoiled
        }

    # For each route, if x>0, we break it into single units
    for (fid, hid, cid, vid), var in x.items():
        qty_val = var.solution_value
        if qty_val > 1e-9:
            # Expand e.g. 5.5 => five single shipments + 0.5 leftover
            whole_part = int(math.floor(qty_val))
            fract_part = qty_val - whole_part

            # Create one record per integer unit
            for _ in range(whole_part):
                route_dict = compute_route_line(fid, hid, cid, vid, 1.0)
                detailed_routes.append(route_dict)

            # If leftover > 0
            if fract_part > 1e-9:
                route_dict = compute_route_line(fid, hid, cid, vid, fract_part)
                detailed_routes.append(route_dict)

    # Sort routes if you want a stable order, e.g. by farm_id, hub_id, center_id
    # (just to have a consistent listing)
    detailed_routes.sort(key=lambda d: (d["farm_id"], d["hub_id"], d["center_id"], d["vehicle_id"]))

    # Final solution dictionary
    solution_dict = {
        "method": "LinearProgramming",
        "best_cost": round(best_cost, 2),
        "total_spoilage_cost": round(total_spoilage_cost, 2),
        "detailed_routes": detailed_routes
    }

    return solution_dict


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example data
    farms_example = [
        {"id": 0, "produce_quantity": 10, "perishability_window": 120},
        {"id": 1, "produce_quantity": 8,  "perishability_window": 100},
        {"id": 2, "produce_quantity": 12, "perishability_window": 150}
    ]
    hubs_example = [
        {"id": 0, "capacity": 15, "storage_cost_per_unit": 2,  "fixed_usage_cost": 50},
        {"id": 1, "capacity": 20, "storage_cost_per_unit": 1.5,"fixed_usage_cost": 60}
    ]
    centers_example = [
        {"id": 0, "demand": 10, "deadline": 130},
        {"id": 1, "demand": 15, "deadline": 140}
    ]
    vehicles_example = [
        {"id": 0, "capacity": 8,  "fixed_cost": 100, "variable_cost_per_distance": 2},
        {"id": 1, "capacity": 10, "fixed_cost": 120, "variable_cost_per_distance": 1.8},
        {"id": 2, "capacity": 12, "fixed_cost": 150, "variable_cost_per_distance": 2.5}
    ]
    dist_dict_example = {
        ("farm", 0, "hub", 0): [{"route_id": 1, "distance_km": 30}],
        ("farm", 0, "hub", 1): [{"route_id": 1, "distance_km": 40}],
        ("farm", 1, "hub", 0): [{"route_id": 1, "distance_km": 35}],
        ("farm", 1, "hub", 1): [{"route_id": 1, "distance_km": 45}],
        ("farm", 2, "hub", 0): [{"route_id": 1, "distance_km": 25}],
        ("farm", 2, "hub", 1): [{"route_id": 1, "distance_km": 30}],
        ("hub", 0, "center", 0): [{"route_id": 1, "distance_km": 50}],
        ("hub", 0, "center", 1): [{"route_id": 1, "distance_km": 45}],
        ("hub", 1, "center", 0): [{"route_id": 1, "distance_km": 55}],
        ("hub", 1, "center", 1): [{"route_id": 1, "distance_km": 40}],
    }

    solution_lp = run_linear_program(
        farms_example,
        hubs_example,
        centers_example,
        vehicles_example,
        dist_dict_example,
        output_file="lp_solution.json"
    )
    print("LP Solution (expanded shipments):")
    print(json.dumps(solution_lp, indent=4))
