#!/usr/bin/env python
"""
linear_programming.py
---------------------
Implements an LP (or MILP) using IBM CPLEX (via docplex) or fallback to PuLP with the CPLEX solver.

Objective:
   Minimize: f(cost) - f(profit)
Where:
   f(cost) = Storage costs + Transport costs + Spoilage losses
   f(profit) = Quantity delivered before deadlines

We handle constraints:
 - Farm produce constraints (can't deliver more than produced).
 - Vehicle capacity constraints.
 - Hub capacity constraints.
 - Perishability/deadline constraints (simplified).
"""

import json
import math
import time

# Try docplex or fallback to pulp
try:
    from docplex.mp.environment import Environment
    from docplex.mp.model import Model
    HAVE_DOCPLEX = True
except ImportError:
    HAVE_DOCPLEX = False
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus, value, LpContinuous


def run_linear_program(farms, hubs, centers, vehicles, dist_dict, output_file="lp_solution.json"):
    """
    Build and solve the LP or MILP. Return a solution dict and also write to JSON.

    For demonstration, we only do single-stage routes (Farm->Hub->Center) in one step with
    continuous decision variables. Real multi-trip, multi-vehicle routing is more complex.

    solution = {
       "method": "LinearProgramming",
       "stage1_assignments": [...],
       "stage2_assignments": [...],
       "objective_value": ...,
       "total_spoilage": ...,
       "delivered_on_time": ...
    }
    """
    start_time = time.time()

    if HAVE_DOCPLEX:
        solution = _run_docplex_lp(farms, hubs, centers, vehicles, dist_dict)
    else:
        solution = _run_pulp_lp(farms, hubs, centers, vehicles, dist_dict)

    solution["runtime_sec"] = round(time.time() - start_time, 2)
    with open(output_file, "w") as f:
        json.dump(solution, f, indent=4)
    return solution


def _run_docplex_lp(farms, hubs, centers, vehicles, dist_dict):
    """
    Using docplex for the LP. Demonstration code. 
    We create flow variables for:
       x[fid, hid, cid, vid] = quantity of farm f produce delivered via hub h to center c by vehicle v
    Then compute cost, spoilage, etc.
    """

    m = Model(name="AgroRouteLP")

    # For simplicity, we define a single super-variable for the entire route (farm->hub->center) with a single vehicle.
    # We also define how to separate or compute cost from that.
    # If dist_fh + dist_hc > perish_window, we consider that produce spoiled -> x=0 forced.

    # Decision variables
    x = {}
    for f in farms:
        fid = f["id"]
        for h in hubs:
            hid = h["id"]
            dist_fh = dist_dict.get(("farm", fid, "hub", hid), None)
            if dist_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                dist_hc = dist_dict.get(("hub", hid, "center", cid), None)
                if dist_hc is None:
                    continue
                for v in vehicles:
                    vid = v["id"]
                    varname = f"x_{fid}_{hid}_{cid}_{vid}"
                    x[(fid,hid,cid,vid)] = m.continuous_var(lb=0, name=varname)

    # Build cost expression
    # f(cost) = sum(transport_cost + storage_cost + spoilage)
    # We'll handle spoilage in a simplified way: if time>perish window => forced x=0 in constraints.
    transport_cost_expr = 0
    storage_cost_expr = 0
    # We'll define a variable for total spoil per farm:
    spoil_vars = {}
    for f in farms:
        spoil_vars[f["id"]] = m.continuous_var(lb=0, name=f"spoil_{f['id']}")

    # We'll define "delivered" as sum of x for each farm. We'll track how much meets the deadline.
    # We can define an indicator if distance sum > deadline => no delivery or partial penalty. 
    # For simplicity, if dist_fh + dist_hc > center.deadline => we treat it as spoiled. We'll enforce x=0.

    # 1) Vehicle capacity constraints
    # sum of x over all f,h,c for each vehicle v <= capacity (one big simplification).
    for v in vehicles:
        vid = v["id"]
        cap = v["capacity"]
        sum_expr = m.sum(x[k] for k in x.keys() if k[3] == vid)
        m.add_constraint(sum_expr <= cap, f"veh_cap_{vid}")

    # 2) Hub capacity constraints
    # sum of x for all farms and centers for a single hub h <= hub.capacity
    for h in hubs:
        hid = h["id"]
        cap = h["capacity"]
        sum_expr = m.sum(x[k] for k in x.keys() if k[1] == hid)
        m.add_constraint(sum_expr <= cap, f"hub_cap_{hid}")

    # 3) Farm produce constraints
    # sum of x across all h,c,v for farm f <= f["produce_quantity"]
    for f in farms:
        fid = f["id"]
        qty = f["produce_quantity"]
        sum_expr = m.sum(x[k] for k in x.keys() if k[0] == fid)
        m.add_constraint(sum_expr + spoil_vars[fid] == qty, f"farm_qty_{fid}")

    # 4) Center demand constraints (we can try to meet or exceed, or treat as cost if missed).
    # For demonstration, let's say we can meet or exceed. No strict requirement here.
    # We won't add a constraint that sum >= demand, but we can track profit from actual delivered portion.
    # 5) Distance-based spoilage: if dist_fh + dist_hc > f.perish_window => x=0 forced.
    for f in farms:
        fid = f["id"]
        pw = f["perishability_window"]
        for h in hubs:
            hid = h["id"]
            dist_fh = dist_dict.get(("farm", fid, "hub", hid), None)
            if dist_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                dist_hc = dist_dict.get(("hub", hid, "center", cid), None)
                if dist_hc is None:
                    continue
                total_dist = dist_fh + dist_hc
                if total_dist > pw:
                    # Force x=0
                    for v in vehicles:
                        vid = v["id"]
                        if (fid,hid,cid,vid) in x:
                            m.add_constraint(x[(fid,hid,cid,vid)] == 0, f"spoil_{fid}_{hid}_{cid}_{vid}")

    # Build cost terms for each x:
    cost_terms = []
    profit_terms = []

    for (fid,hid,cid,vid) in x:
        fobj = next(ff for ff in farms if ff["id"] == fid)
        hobj = next(hh for hh in hubs if hh["id"] == hid)
        cobj = next(cc for cc in centers if cc["id"] == cid)
        vobj = next(vv for vv in vehicles if vv["id"] == vid)

        dist_fh = dist_dict[("farm", fid, "hub", hid)]
        dist_hc = dist_dict[("hub", hid, "center", cid)]
        # Transport cost: vehicle fixed + variable * distance?
        # We'll approximate: cost = fixed_cost + (dist_fh+dist_hc)*var_cost, but that's if the vehicle is used only once
        # More advanced modeling would have binary usage variables. Let's do a naive proportion approach:
        # We'll do: cost = x/(vehicle capacity) * [fixed_cost + (dist_fh+dist_hc)*var_cost]
        # This is a common "fractional usage" assumption (not 100% correct for integral VRP).
        dist_total = dist_fh + dist_hc
        cost_per_unit = (vobj["fixed_cost"] + dist_total * vobj["variable_cost_per_distance"]) / vobj["capacity"]

        # Storage cost: x * hobj["storage_cost_per_unit"] (assuming 1 time unit) + fixed_usage_cost if x>0
        # We'll similarly do a fraction: cost for usage = x/(hub capacity)*hub.fixed_usage_cost
        # + x*hobj.storage_cost_per_unit
        store_cost_per_unit = hobj["storage_cost_per_unit"] + (hobj["fixed_usage_cost"] / max(1,hobj["capacity"]))

        # Build expression
        cost_expr = x[(fid,hid,cid,vid)] * (cost_per_unit + store_cost_per_unit)
        cost_terms.append(cost_expr)

        # Profit: if total_dist <= cobj["deadline"], we count it as on-time
        if dist_total <= cobj["deadline"]:
            # Profit = x units delivered
            profit_terms.append(x[(fid,hid,cid,vid)])
        else:
            profit_terms.append(0)

    # Spoilage cost: we define a simple cost per spoil, say 2 units cost for each produce spoiled
    spoil_cost_expr = 0
    for f in farms:
        spoil_cost_expr += 2 * spoil_vars[f["id"]]

    # Objective = sum(cost) + sum(spoil_cost) - sum(profit)
    total_cost_expr = m.sum(cost_terms) + spoil_cost_expr
    total_profit_expr = m.sum(profit_terms)
    # We want to minimize (cost_expr - profit_expr)
    obj_expr = total_cost_expr - total_profit_expr

    m.minimize(obj_expr)

    sol = m.solve(log_output=False)
    if sol is None:
        return {
            "method": "LinearProgramming(docplex)",
            "status": "No feasible solution",
            "stage1_assignments": [],
            "stage2_assignments": [],
            "objective_value": None,
            "total_spoilage": None,
            "delivered_on_time": 0
        }

    # Build solution output
    stage1 = []
    stage2 = []
    total_spoil = 0
    for f in farms:
        total_spoil += spoil_vars[f["id"]].solution_value
    delivered_on_time = 0
    for (fid,hid,cid,vid) in x:
        qty = x[(fid,hid,cid,vid)].solution_value
        if qty > 1e-6:
            # We treat stage1 = farm->hub, stage2 = hub->center in aggregated form
            stage1.append({
                "farm_id": fid,
                "hub_id": hid,
                "vehicle_id": vid,
                "quantity": qty
            })
            stage2.append({
                "hub_id": hid,
                "center_id": cid,
                "vehicle_id": vid,
                "quantity": qty
            })
            fobj = next(ff for ff in farms if ff["id"] == fid)
            cobj = next(cc for cc in centers if cc["id"] == cid)
            dist_fh = dist_dict[("farm", fid, "hub", hid)]
            dist_hc = dist_dict[("hub", hid, "center", cid)]
            if (dist_fh + dist_hc) <= cobj["deadline"]:
                delivered_on_time += qty

    solution = {
        "method": "LinearProgramming(docplex)",
        "status": str(sol.solve_status),
        "objective_value": sol.objective_value,
        "stage1_assignments": stage1,
        "stage2_assignments": stage2,
        "total_spoilage": total_spoil,
        "delivered_on_time": delivered_on_time
    }
    return solution


def _run_pulp_lp(farms, hubs, centers, vehicles, dist_dict):
    """
    Fallback using PuLP, configured to use CPLEX if available. Similar logic to docplex version.
    """

    model = LpProblem("AgroRouteLP", LpMinimize)

    # Decision variables
    x = {}
    for f in farms:
        fid = f["id"]
        for h in hubs:
            hid = h["id"]
            dist_fh = dist_dict.get(("farm", fid, "hub", hid), None)
            if dist_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                dist_hc = dist_dict.get(("hub", hid, "center", cid), None)
                if dist_hc is None:
                    continue
                for v in vehicles:
                    vid = v["id"]
                    varname = f"x_{fid}_{hid}_{cid}_{vid}"
                    x[(fid,hid,cid,vid)] = LpVariable(varname, lowBound=0, cat="Continuous")

    # Spoilage variables
    spoil_vars = {}
    for f in farms:
        fid = f["id"]
        spoil_vars[fid] = LpVariable(f"spoil_{fid}", lowBound=0, cat="Continuous")

    # Capacity constraints
    for v in vehicles:
        vid = v["id"]
        cap = v["capacity"]
        model += lpSum(x[k] for k in x.keys() if k[3] == vid) <= cap, f"veh_cap_{vid}"

    for h in hubs:
        hid = h["id"]
        cap = h["capacity"]
        model += lpSum(x[k] for k in x.keys() if k[1] == hid) <= cap, f"hub_cap_{hid}"

    for f in farms:
        fid = f["id"]
        qty = f["produce_quantity"]
        model += lpSum(x[k] for k in x.keys() if k[0] == fid) + spoil_vars[fid] == qty, f"farm_qty_{fid}"

    # Distance-based spoil
    for f in farms:
        fid = f["id"]
        pw = f["perishability_window"]
        for h in hubs:
            hid = h["id"]
            dist_fh = dist_dict.get(("farm", fid, "hub", hid), None)
            if dist_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                dist_hc = dist_dict.get(("hub", hid, "center", cid), None)
                if dist_hc is None:
                    continue
                total_dist = dist_fh + dist_hc
                if total_dist > pw:
                    for v in vehicles:
                        vid = v["id"]
                        if (fid,hid,cid,vid) in x:
                            model += x[(fid,hid,cid,vid)] == 0

    # Cost + Profit
    cost_terms = []
    profit_terms = []
    for (fid,hid,cid,vid), var in x.items():
        fobj = next(ff for ff in farms if ff["id"] == fid)
        hobj = next(hh for hh in hubs if hh["id"] == hid)
        cobj = next(cc for cc in centers if cc["id"] == cid)
        vobj = next(vv for vv in vehicles if vv["id"] == vid)

        dist_fh = dist_dict[("farm", fid, "hub", hid)]
        dist_hc = dist_dict[("hub", hid, "center", cid)]
        dist_total = dist_fh + dist_hc
        # Simplify cost
        cost_per_unit = (vobj["fixed_cost"] + dist_total * vobj["variable_cost_per_distance"]) / vobj["capacity"]
        store_cost_per_unit = hobj["storage_cost_per_unit"] + (hobj["fixed_usage_cost"] / max(1,hobj["capacity"]))

        cost_expr = var * (cost_per_unit + store_cost_per_unit)
        cost_terms.append(cost_expr)

        # Profit if total_dist <= cobj.deadline
        if dist_total <= cobj["deadline"]:
            profit_terms.append(var)
        else:
            profit_terms.append(0)

    spoil_cost_expr = lpSum(2 * spoil_vars[f["id"]] for f in farms)

    total_cost_expr = lpSum(cost_terms) + spoil_cost_expr
    total_profit_expr = lpSum(profit_terms)

    model += total_cost_expr - total_profit_expr, "Objective"

    model_status = model.solve()  # if cplex is installed, can do model.solve(pulp.CPLEX_PY())
    status_str = LpStatus[model.status]
    if status_str not in ["Optimal", " feasible"]:
        return {
            "method": "LinearProgramming(PuLP+CPLEX)",
            "status": status_str,
            "stage1_assignments": [],
            "stage2_assignments": [],
            "objective_value": None,
            "total_spoilage": None,
            "delivered_on_time": 0
        }

    # Gather solution
    stage1 = []
    stage2 = []
    total_spoil = 0
    for f in farms:
        total_spoil += spoil_vars[f["id"]].varValue

    delivered_on_time = 0
    for (fid,hid,cid,vid), var in x.items():
        qty = var.varValue
        if qty > 1e-6:
            stage1.append({
                "farm_id": fid,
                "hub_id": hid,
                "vehicle_id": vid,
                "quantity": qty
            })
            stage2.append({
                "hub_id": hid,
                "center_id": cid,
                "vehicle_id": vid,
                "quantity": qty
            })
            cobj = next(cc for cc in centers if cc["id"] == cid)
            dist_fh = dist_dict[("farm", fid, "hub", hid)]
            dist_hc = dist_dict[("hub", hid, "center", cid)]
            if (dist_fh + dist_hc) <= cobj["deadline"]:
                delivered_on_time += qty

    solution = {
        "method": "LinearProgramming(PuLP+CPLEX)",
        "status": status_str,
        "objective_value": value(model.objective),
        "stage1_assignments": stage1,
        "stage2_assignments": stage2,
        "total_spoilage": total_spoil,
        "delivered_on_time": delivered_on_time
    }
    return solution
