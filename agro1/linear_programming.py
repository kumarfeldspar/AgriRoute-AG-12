#!/usr/bin/env python
"""
linear_programming.py
---------------------
A demonstration of an LP-based approach. We define x[f,h,c,v] = quantity farm->hub->center
and minimize cost - profit.

We now store more details about each route in the final solution:
 - distance farm->hub, distance hub->center
 - cost for that route
 - partial breakdown, etc.
"""

import json
import math
import time

# Attempt docplex or fallback to pulp
try:
    from docplex.mp.model import Model
    HAVE_DOCPLEX = True
except ImportError:
    HAVE_DOCPLEX = False
    from pulp import (LpProblem, LpMinimize, LpVariable, lpSum, LpContinuous, LpStatus, value)

BIG_M = 999999

def _pick_primary_distance(dist_list):
    """
    dist_list = [ {route_id:1, distance_km:x}, {route_id:2, ...} ]
    We'll pick route_id=1 if it exists.
    """
    if not dist_list:
        return None
    for rd in dist_list:
        if rd["route_id"] == 1:
            return rd["distance_km"]
    return None

def run_linear_program(farms, hubs, centers, vehicles, dist_dict, output_file="lp_solution.json"):
    start_time = time.time()
    if HAVE_DOCPLEX:
        solution = _run_docplex_lp(farms, hubs, centers, vehicles, dist_dict)
    else:
        solution = _run_pulp_lp(farms, hubs, centers, vehicles, dist_dict)

    solution["runtime_sec"] = round(time.time() - start_time, 2)

    # Dump to file
    with open(output_file, "w") as f:
        json.dump(solution, f, indent=4)

    return solution

def _run_docplex_lp(farms, hubs, centers, vehicles, dist_dict):
    m = Model(name="AgroRouteLP_docplex")

    # x[f,h,c,v] = quantity
    x = {}
    # spoil_f = how much spoils from farm f
    spoil_f = {}

    for f in farms:
        fid = f["id"]
        spoil_f[fid] = m.continuous_var(lb=0, name=f"spoil_{fid}")

    for f in farms:
        fid = f["id"]
        for h in hubs:
            hid = h["id"]
            roads_fh = dist_dict.get(("farm", fid, "hub", hid), [])
            d_fh = _pick_primary_distance(roads_fh)
            if d_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                roads_hc = dist_dict.get(("hub", hid, "center", cid), [])
                d_hc = _pick_primary_distance(roads_hc)
                if d_hc is None:
                    continue
                for v in vehicles:
                    vid = v["id"]
                    var = m.continuous_var(lb=0, name=f"x_{fid}_{hid}_{cid}_{vid}")
                    x[(fid,hid,cid,vid)] = var

    # Constraints

    # 1) Vehicle capacity: sum of x over all f,h,c for each vehicle v <= capacity
    for v in vehicles:
        vid = v["id"]
        cap = v["capacity"]
        sum_expr = m.sum(x[k] for k in x if k[3] == vid)
        m.add_constraint(sum_expr <= cap, f"veh_cap_{vid}")

    # 2) Hub capacity
    for h in hubs:
        hid = h["id"]
        cap = h["capacity"]
        sum_expr = m.sum(x[k] for k in x if k[1] == hid)
        m.add_constraint(sum_expr <= cap, f"hub_cap_{hid}")

    # 3) Farm produce: sum of x + spoil = produce_quantity
    for f in farms:
        fid = f["id"]
        qty = f["produce_quantity"]
        sum_expr = m.sum(x[k] for k in x if k[0] == fid)
        m.add_constraint(sum_expr + spoil_f[fid] == qty, f"farm_qty_{fid}")

    # 4) Perishability constraint: if distance > window => x=0
    for f in farms:
        fid = f["id"]
        pw = f["perishability_window"]
        for h in hubs:
            hid = h["id"]
            roads_fh = dist_dict.get(("farm", fid, "hub", hid), [])
            d_fh = _pick_primary_distance(roads_fh)
            if d_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                roads_hc = dist_dict.get(("hub", hid, "center", cid), [])
                d_hc = _pick_primary_distance(roads_hc)
                if d_hc is None:
                    continue
                total_d = d_fh + d_hc
                if total_d > pw:
                    for v in vehicles:
                        vid = v["id"]
                        if (fid,hid,cid,vid) in x:
                            m.add_constraint(x[(fid,hid,cid,vid)] == 0)

    # Build cost and profit expressions
    cost_terms = []
    profit_terms = []

    def storage_cost_per_unit(hub):
        # approximate: storage cost + fraction of fixed usage
        return hub["storage_cost_per_unit"] + (hub["fixed_usage_cost"] / max(1, hub["capacity"]))

    # Spoilage cost: e.g. 2 currency per unit spoiled
    spoil_cost_expr = m.sum(2 * spoil_f[f["id"]] for f in farms)

    for (fid,hid,cid,vid), var in x.items():
        fobj = next(ff for ff in farms if ff["id"] == fid)
        hobj = next(hh for hh in hubs if hh["id"] == hid)
        cobj = next(cc for cc in centers if cc["id"] == cid)
        vobj = next(vv for vv in vehicles if vv["id"] == vid)

        # distances:
        d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
        d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
        if d_fh is None or d_hc is None:
            continue

        dist_total = d_fh + d_hc
        # transport cost fraction
        cost_per_unit = (vobj["fixed_cost"] + dist_total * vobj["variable_cost_per_distance"]) / vobj["capacity"]
        # storage
        store_per_unit = storage_cost_per_unit(hobj)
        cost_expr = var * (cost_per_unit + store_per_unit)
        cost_terms.append(cost_expr)

        # if dist_total <= cobj["deadline"], we get profit = var
        if dist_total <= cobj["deadline"]:
            profit_terms.append(var)
        else:
            profit_terms.append(0)

    total_cost_expr = m.sum(cost_terms) + spoil_cost_expr
    total_profit_expr = m.sum(profit_terms)
    # Minimize cost - profit
    m.minimize(total_cost_expr - total_profit_expr)

    sol = m.solve(log_output=False)
    if sol is None:
        return {
            "method": "LinearProgramming(docplex)",
            "status": "No feasible solution",
            "objective_value": None,
            "detailed_routes": [],
            "total_spoilage": None,
            "delivered_on_time": 0
        }

    # Build solution
    stage_routes = []
    total_spoil = sum(spoil_f[f["id"]].solution_value for f in farms)
    delivered_on_time = 0
    for (fid,hid,cid,vid), var in x.items():
        qty = var.solution_value
        if qty > 1e-6:
            # gather info
            fobj = next(ff for ff in farms if ff["id"] == fid)
            hobj = next(hh for hh in hubs if hh["id"] == hid)
            cobj = next(cc for cc in centers if cc["id"] == cid)
            vobj = next(vv for vv in vehicles if vv["id"] == vid)

            d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
            d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
            dist_total = d_fh + d_hc
            # cost
            cost_pu = (vobj["fixed_cost"] + dist_total * vobj["variable_cost_per_distance"]) / vobj["capacity"]
            store_pu = (hobj["storage_cost_per_unit"] + (hobj["fixed_usage_cost"]/max(1,hobj["capacity"])))
            route_cost = qty*(cost_pu + store_pu)
            on_time = (1 if dist_total <= cobj["deadline"] else 0)
            if on_time:
                delivered_on_time += qty

            stage_routes.append({
                "farm_id": fid,
                "hub_id": hid,
                "center_id": cid,
                "vehicle_id": vid,
                "quantity": qty,
                "dist_farm_hub": d_fh,
                "dist_hub_center": d_hc,
                "total_dist": dist_total,
                "route_cost": round(route_cost, 3),
                "on_time": bool(on_time)
            })

    solution = {
        "method": "LinearProgramming(docplex)",
        "status": str(sol.solve_status),  # convert to string
        "objective_value": sol.objective_value,
        "detailed_routes": stage_routes,
        "total_spoilage": round(total_spoil, 2),
        "delivered_on_time": round(delivered_on_time, 2)
    }
    return solution

def _run_pulp_lp(farms, hubs, centers, vehicles, dist_dict):
    model = LpProblem("AgroRouteLP_pulp", LpMinimize)

    x = {}
    spoil_vars = {}

    for f in farms:
        fid = f["id"]
        spoil_vars[fid] = LpVariable(f"spoil_{fid}", lowBound=0, cat="Continuous")

    for f in farms:
        fid = f["id"]
        for h in hubs:
            hid = h["id"]
            roads_fh = dist_dict.get(("farm", fid, "hub", hid), [])
            d_fh = _pick_primary_distance(roads_fh)
            if d_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                roads_hc = dist_dict.get(("hub", hid, "center", cid), [])
                d_hc = _pick_primary_distance(roads_hc)
                if d_hc is None:
                    continue
                for v in vehicles:
                    vid = v["id"]
                    varname = f"x_{fid}_{hid}_{cid}_{vid}"
                    x[(fid,hid,cid,vid)] = LpVariable(varname, lowBound=0, cat="Continuous")

    # capacity constraints
    for v in vehicles:
        vid = v["id"]
        cap = v["capacity"]
        model += lpSum(x[k] for k in x if k[3] == vid) <= cap, f"veh_cap_{vid}"

    for h in hubs:
        hid = h["id"]
        cap = h["capacity"]
        model += lpSum(x[k] for k in x if k[1] == hid) <= cap, f"hub_cap_{hid}"

    for f in farms:
        fid = f["id"]
        qty = f["produce_quantity"]
        model += lpSum(x[k] for k in x if k[0] == fid) + spoil_vars[fid] == qty, f"farm_qty_{fid}"

    # perish
    for f in farms:
        fid = f["id"]
        pw = f["perishability_window"]
        for h in hubs:
            hid = h["id"]
            d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
            if d_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
                if d_hc is None:
                    continue
                total_dist = 0 if (d_fh is None or d_hc is None) else d_fh + d_hc
                if total_dist > pw:
                    for v in vehicles:
                        vid = v["id"]
                        if (fid,hid,cid,vid) in x:
                            model += x[(fid,hid,cid,vid)] == 0

    # build cost - profit
    cost_terms = []
    profit_terms = []
    for (fid,hid,cid,vid), var in x.items():
        fobj = next(ff for ff in farms if ff["id"] == fid)
        hobj = next(hh for hh in hubs if hh["id"] == hid)
        cobj = next(cc for cc in centers if cc["id"] == cid)
        vobj = next(vv for vv in vehicles if vv["id"] == vid)

        d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
        d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
        dist_total = d_fh + d_hc
        cost_pu = (vobj["fixed_cost"] + dist_total*vobj["variable_cost_per_distance"]) / vobj["capacity"]
        store_pu = (hobj["storage_cost_per_unit"] + hobj["fixed_usage_cost"]/max(1,hobj["capacity"]))
        cost_expr = var*(cost_pu + store_pu)
        cost_terms.append(cost_expr)

        if dist_total <= cobj["deadline"]:
            profit_terms.append(var)
        else:
            profit_terms.append(0)

    spoil_cost_expr = lpSum(2*spoil_vars[f["id"]] for f in farms)

    model += lpSum(cost_terms) + spoil_cost_expr - lpSum(profit_terms), "Objective"

    status = model.solve()
    status_str = LpStatus[model.status]
    if status_str not in ["Optimal", "Feasible"]:
        return {
            "method": "LinearProgramming(PuLP)",
            "status": status_str,
            "objective_value": None,
            "detailed_routes": [],
            "total_spoilage": None,
            "delivered_on_time": 0
        }

    # build solution
    stage_routes = []
    total_spoil = sum(spoil_vars[f["id"]].varValue for f in farms)
    delivered_on_time = 0
    for (fid,hid,cid,vid), var in x.items():
        qty = var.varValue
        if qty > 1e-6:
            fobj = next(ff for ff in farms if ff["id"] == fid)
            hobj = next(hh for hh in hubs if hh["id"] == hid)
            cobj = next(cc for cc in centers if cc["id"] == cid)
            vobj = next(vv for vv in vehicles if vv["id"] == vid)

            d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
            d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
            dist_total = d_fh + d_hc
            cost_pu = (vobj["fixed_cost"] + dist_total*vobj["variable_cost_per_distance"]) / vobj["capacity"]
            store_pu = (hobj["storage_cost_per_unit"] + hobj["fixed_usage_cost"]/max(1,hobj["capacity"]))
            route_cost = qty*(cost_pu + store_pu)
            on_time = (dist_total <= cobj["deadline"])
            if on_time:
                delivered_on_time += qty

            stage_routes.append({
                "farm_id": fid,
                "hub_id": hid,
                "center_id": cid,
                "vehicle_id": vid,
                "quantity": qty,
                "dist_farm_hub": d_fh,
                "dist_hub_center": d_hc,
                "total_dist": dist_total,
                "route_cost": round(route_cost, 3),
                "on_time": on_time
            })

    solution = {
        "method": "LinearProgramming(PuLP)",
        "status": status_str,
        "objective_value": value(model.objective),
        "detailed_routes": stage_routes,
        "total_spoilage": round(total_spoil,2),
        "delivered_on_time": round(delivered_on_time,2)
    }
    return solution
