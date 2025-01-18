#!/usr/bin/env python
"""
linear_programming.py
---------------------
A demonstration of an LP-based approach. We define x[f,h,c,v] = quantity farm->hub->center
and minimize cost - profit, subject to constraints.

We return a solution with "detailed_routes" containing route-level info.
"""

import json
import time

# Attempt docplex or fallback to pulp
try:
    from docplex.mp.model import Model
    HAVE_DOCPLEX = True
except ImportError:
    HAVE_DOCPLEX = False
    from pulp import (LpProblem, LpMinimize, LpVariable, lpSum, LpContinuous, LpStatus, value)

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

    with open(output_file, "w") as f:
        json.dump(solution, f, indent=4)

    return solution

def _run_docplex_lp(farms, hubs, centers, vehicles, dist_dict):
    from docplex.mp.model import Model
    m = Model(name="AgroRouteLP_docplex")

    # x[f,h,c,v] = quantity
    x = {}
    # spoil_f[f] = how much spoils from farm f
    spoil_f = {}

    for f in farms:
        fid = f["id"]
        spoil_f[fid] = m.continuous_var(lb=0, name=f"spoil_{fid}")

    for f in farms:
        fid = f["id"]
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
                for v in vehicles:
                    vid = v["id"]
                    var = m.continuous_var(lb=0, name=f"x_{fid}_{hid}_{cid}_{vid}")
                    x[(fid,hid,cid,vid)] = var

    # constraints

    # 1) vehicle capacity
    for v in vehicles:
        vid = v["id"]
        cap = v["capacity"]
        m.add_constraint(m.sum(x[k] for k in x if k[3] == vid) <= cap, f"veh_cap_{vid}")

    # 2) hub capacity
    for h in hubs:
        hid = h["id"]
        cap = h["capacity"]
        m.add_constraint(m.sum(x[k] for k in x if k[1] == hid) <= cap, f"hub_cap_{hid}")

    # 3) farm produce
    for f in farms:
        fid = f["id"]
        qty = f["produce_quantity"]
        m.add_constraint(m.sum(x[k] for k in x if k[0] == fid) + spoil_f[fid] == qty, f"farm_qty_{fid}")

    # 4) perishability
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
                total_d = d_fh + d_hc
                if total_d > pw:
                    for v in vehicles:
                        vid = v["id"]
                        if (fid,hid,cid,vid) in x:
                            m.add_constraint(x[(fid,hid,cid,vid)] == 0)

    # cost-profit
    cost_terms = []
    profit_terms = []
    # spoil cost
    spoil_cost_expr = m.sum(2 * spoil_f[f["id"]] for f in farms)

    def storage_cost_per_unit(hub):
        return hub["storage_cost_per_unit"] + (hub["fixed_usage_cost"] / max(1, hub["capacity"]))

    for (fid,hid,cid,vid), var in x.items():
        fobj = next(ff for ff in farms if ff["id"] == fid)
        hobj = next(hh for hh in hubs if hh["id"] == hid)
        cobj = next(cc for cc in centers if cc["id"] == cid)
        vobj = next(vv for vv in vehicles if vv["id"] == vid)
        d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
        d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
        dist_total = d_fh + d_hc

        cost_pu = (vobj["fixed_cost"] + dist_total*vobj["variable_cost_per_distance"]) / vobj["capacity"]
        store_pu = storage_cost_per_unit(hobj)
        cost_expr = var * (cost_pu + store_pu)
        cost_terms.append(cost_expr)

        # if dist_total <= cobj["deadline"], profit = var
        if dist_total <= cobj["deadline"]:
            profit_terms.append(var)
        else:
            profit_terms.append(0)

    total_cost_expr = m.sum(cost_terms) + spoil_cost_expr
    total_profit_expr = m.sum(profit_terms)

    # minimize cost - profit
    m.minimize(total_cost_expr - total_profit_expr)

    sol = m.solve(log_output=False)
    if not sol:
        return {
            "method": "LinearProgramming(docplex)",
            "status": "No feasible solution",
            "objective_value": None,
            "detailed_routes": [],
            "total_spoilage": None,
            "delivered_on_time": 0
        }

    # build solution
    stage_routes = []
    total_spoil = sum(spoil_f[f["id"]].solution_value for f in farms)
    delivered_on_time = 0
    for (fid,hid,cid,vid), var in x.items():
        qty = var.solution_value
        if qty > 1e-6:
            fobj = next(ff for ff in farms if ff["id"] == fid)
            hobj = next(hh for hh in hubs if hh["id"] == hid)
            cobj = next(cc for cc in centers if cc["id"] == cid)
            vobj = next(vv for vv in vehicles if vv["id"] == vid)
            d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
            d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
            dist_total = d_fh + d_hc
            route_cost = qty * ((vobj["fixed_cost"] + dist_total*vobj["variable_cost_per_distance"]) / vobj["capacity"]
                                + (hobj["storage_cost_per_unit"] + (hobj["fixed_usage_cost"]/max(1,hobj["capacity"]))))

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

    return {
        "method": "LinearProgramming(docplex)",
        "status": str(sol.solve_status),
        "objective_value": sol.objective_value,
        "detailed_routes": stage_routes,
        "total_spoilage": round(total_spoil, 2),
        "delivered_on_time": round(delivered_on_time, 2)
    }

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
            d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
            if d_fh is None:
                continue
            for c in centers:
                cid = c["id"]
                d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
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
                total_dist = d_fh + d_hc
                if total_dist > pw:
                    for v in vehicles:
                        vid = v["id"]
                        if (fid,hid,cid,vid) in x:
                            model += x[(fid,hid,cid,vid)] == 0

    cost_terms = []
    profit_terms = []
    spoil_cost_expr = lpSum(2*spoil_vars[f["id"]] for f in farms)

    def storage_cost_per_unit(hub):
        return hub["storage_cost_per_unit"] + (hub["fixed_usage_cost"] / max(1, hub["capacity"]))

    for (fid,hid,cid,vid), var in x.items():
        fobj = next(ff for ff in farms if ff["id"] == fid)
        hobj = next(hh for hh in hubs if hh["id"] == hid)
        cobj = next(cc for cc in centers if cc["id"] == cid)
        vobj = next(vv for vv in vehicles if vv["id"] == vid)
        d_fh = _pick_primary_distance(dist_dict.get(("farm", fid, "hub", hid), []))
        d_hc = _pick_primary_distance(dist_dict.get(("hub", hid, "center", cid), []))
        dist_total = d_fh + d_hc

        cost_pu = (vobj["fixed_cost"] + dist_total*vobj["variable_cost_per_distance"]) / vobj["capacity"]
        store_pu = storage_cost_per_unit(hobj)
        cost_expr = var*(cost_pu + store_pu)
        cost_terms.append(cost_expr)

        if dist_total <= cobj["deadline"]:
            profit_terms.append(var)
        else:
            profit_terms.append(0)

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
            store_pu = storage_cost_per_unit(hobj)
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

    return {
        "method": "LinearProgramming(PuLP)",
        "status": status_str,
        "objective_value": value(model.objective),
        "detailed_routes": stage_routes,
        "total_spoilage": round(total_spoil,2),
        "delivered_on_time": round(delivered_on_time,2)
    }
