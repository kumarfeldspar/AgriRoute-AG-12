#!/usr/bin/env python
"""
novel_optimization.py
---------------------
A simplified demonstration of a "novel" approach using MILP.

We define decision variables for:
  x_{f,h,v} = 1 if farm f is served by vehicle v and goes to hub h
  y_{h,c,v} = 1 if hub h is served by vehicle v going to center c
We also incorporate capacity constraints for hubs and vehicles, plus disruptions, etc.

Outputs:
  novel_solution.json
"""

import json
import math
import random
import time
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus, value

def distance(lat1, lon1, lat2, lon2):
    # Euclidean for simplicity
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def main():
    with open("farms.json") as f:
        farms = json.load(f)
    with open("hubs.json") as f:
        hubs = json.load(f)
    with open("centers.json") as f:
        centers = json.load(f)
    with open("vehicles.json") as f:
        vehicles = json.load(f)
    with open("disruptions.json") as f:
        disruptions = json.load(f)

    # Create a quick map for disruptions
    disruption_map = {}
    for d in disruptions:
        start = d["start"]  # (type, id)
        end = d["end"]
        disruption_map[(start[0], start[1], end[0], end[1])] = {
            "closed": d["closed"],
            "cost_mult": d["cost_multiplier"]
        }

    # Build cost / distance data for farm->hub and hub->center
    farm_hub_dist = {}
    farm_hub_cost_mult = {}
    for fobj in farms:
        fid = fobj["id"]
        fx, fy = fobj["location"]
        for hobj in hubs:
            hid = hobj["id"]
            hx, hy = hobj["location"]
            dist = distance(fx, fy, hx, hy)
            key = (fid, hid)
            # Disruption
            dis = disruption_map.get(("farm", fid, "hub", hid), {"closed": False, "cost_mult": 1.0})
            if dis["closed"]:
                farm_hub_dist[key] = None  # mark as no route
            else:
                farm_hub_dist[key] = dist
            farm_hub_cost_mult[key] = dis["cost_mult"]

    hub_center_dist = {}
    hub_center_cost_mult = {}
    for hobj in hubs:
        hid = hobj["id"]
        hx, hy = hobj["location"]
        for cobj in centers:
            cid = cobj["id"]
            cx, cy = cobj["location"]
            dist = distance(hx, hy, cx, cy)
            key = (hid, cid)
            dis = disruption_map.get(("hub", hid, "center", cid), {"closed": False, "cost_mult": 1.0})
            if dis["closed"]:
                hub_center_dist[key] = None
            else:
                hub_center_dist[key] = dist
            hub_center_cost_mult[key] = dis["cost_mult"]

    # Start building MILP model
    model = LpProblem("MultiStageAgriSupply", LpMinimize)

    # Decision variables
    # x[(f,h,v)] = 1 if farm f's produce is assigned to hub h by vehicle v
    x = {}
    # y[(h,c,v)] = 1 if hub h is assigned to center c by vehicle v
    y = {}

    # We also need to handle how much quantity is shipped farm->hub
    # to account for partial shipments if needed
    qfhv = {}

    # Similarly quantity from hub->center
    qhcv = {}

    # Create variables
    for fobj in farms:
        fid = fobj["id"]
        fqty = fobj["quantity"]
        for hobj in hubs:
            hid = hobj["id"]
            if farm_hub_dist.get((fid, hid)) is None:
                # route closed
                continue
            for vobj in vehicles:
                vid = vobj["id"]
                varname = f"x_{fid}_{hid}_{vid}"
                x[(fid,hid,vid)] = LpVariable(varname, cat=LpBinary)

                # continuous variable for quantity
                qname = f"qfhv_{fid}_{hid}_{vid}"
                qfhv[(fid,hid,vid)] = LpVariable(qname, lowBound=0, upBound=fqty, cat="Continuous")

    for hobj in hubs:
        hid = hobj["id"]
        for cobj in centers:
            cid = cobj["id"]
            if hub_center_dist.get((hid, cid)) is None:
                continue
            for vobj in vehicles:
                vid = vobj["id"]
                varname = f"y_{hid}_{cid}_{vid}"
                y[(hid,cid,vid)] = LpVariable(varname, cat=LpBinary)

                qname = f"qhcv_{hid}_{cid}_{vid}"
                # upper bound = sum of possible farm produce, but let's just pick a large number
                qhcv[(hid,cid,vid)] = LpVariable(qname, lowBound=0, cat="Continuous")

    # Objective: Minimize total cost + spoilage penalty
    # cost = sum over farm->hub of (distance * cost_per_km(v) * cost_mult + hub usage cost if used) + ...
    # plus a spoilage penalty if not delivered within perish_window, etc.
    # We'll do a simplified approach: if any produce is assigned from f->h with distance d, cost = d * cost_per_km * cost_mult
    # Then from h->c likewise. Also add hub usage cost if we route >0 from that hub. Then add penalty for produce not delivered.
    # not_delivered = farm's quantity - sum of qfhv from that farm => penalty

    bigM = 999999
    cost_expr = []

    # farm->hub cost
    for (fid,hid,vid), var in x.items():
        dist = farm_hub_dist[(fid,hid)]
        cost_mult = farm_hub_cost_mult[(fid,hid)]
        v = next(vv for vv in vehicles if vv["id"] == vid)
        cost_expr.append(dist * v["cost_per_km"] * cost_mult * var)

    # hub->center cost
    for (hid,cid,vid), var in y.items():
        dist = hub_center_dist[(hid,cid)]
        cost_mult = hub_center_cost_mult[(hid,cid)]
        v = next(vv for vv in vehicles if vv["id"] == vid)
        cost_expr.append(dist * v["cost_per_km"] * cost_mult * var)

    # usage cost for each hub if used at all
    # if sum(x for that hub) > 0 => pay usage cost
    for hobj in hubs:
        hid = hobj["id"]
        usage = hobj["usage_cost"]
        # define an auxiliary variable: hub_used[h] in {0,1}
        hub_used_var = LpVariable(f"hub_used_{hid}", cat=LpBinary)
        # link constraints: hub_used >= x[(f,h,v)] for all f,v
        for fobj in farms:
            fid = fobj["id"]
            for vobj in vehicles:
                vid = vobj["id"]
                if (fid,hid,vid) in x:
                    model += hub_used_var >= x[(fid,hid,vid)]
        # cost contribution
        cost_expr.append(usage * hub_used_var)

    # spoilage penalty
    # let delivered_f = sum of qfhv for that farm across all h,v
    # if delivered_f < farm quantity => spoil
    # but also time constraints with perish_window => we do a simple approach
    # we'll define a "spoil_f" variable to represent how much is spoiled from farm f
    spoil_vars = {}
    for fobj in farms:
        fid = fobj["id"]
        qvar = LpVariable(f"spoil_{fid}", lowBound=0, cat="Continuous")
        spoil_vars[fid] = qvar

    penalty = 5.0  # cost per unit of spoiled produce (example)
    for fid, spoil_var in spoil_vars.items():
        cost_expr.append(penalty * spoil_var)

    # Add objective
    model += lpSum(cost_expr)

    # Constraints

    # 1) Vehicle capacity constraints (farm->hub):
    # sum of qfhv for each vehicle v cannot exceed vehicle capacity
    # also tie qfhv <= x*(farm quantity) to ensure if x=0 => no flow
    for (fid,hid,vid) in x.keys():
        fqty = next(f["quantity"] for f in farms if f["id"] == fid)
        model += qfhv[(fid,hid,vid)] <= x[(fid,hid,vid)] * fqty
    # sum of qfhv for each vehicle v on a single trip shouldn't exceed capacity
    # (this is a simplification since in reality you'd have route-based constraints,
    # but we'll treat it as if each vehicle can only do one route from a farm to a hub)
    # For demonstration:
    for vid in set(v["id"] for v in vehicles):
        vcap = next(vv["capacity"] for vv in vehicles if vv["id"] == vid)
        # sum over all farms,hubs
        model += lpSum(qfhv[(fid,hid,vid)] for (fid,hid,vv) in x.keys() if vv==vid) <= vcap

    # 2) Vehicle capacity constraints (hub->center):
    for (hid,cid,vid) in y.keys():
        # if y=0 => qhcv=0
        model += qhcv[(hid,cid,vid)] <= y[(hid,cid,vid)] * bigM

    for vid in set(v["id"] for v in vehicles):
        vcap = next(vv["capacity"] for vv in vehicles if vv["id"] == vid)
        # sum over all hubs,centers
        model += lpSum(qhcv[(hid,cid,vid)] for (hid,cid,vv) in y.keys() if vv==vid) <= vcap

    # 3) Hub capacity: sum of all incoming produce cannot exceed hub capacity
    for hobj in hubs:
        hid = hobj["id"]
        cap = hobj["capacity"]
        # sum of qfhv for that hub over all farms, vehicles
        model += lpSum(qfhv[(fid,hid,vid)] for (fid,hid,vid) in x.keys() if hid==hobj["id"]) <= cap

    # 4) Demand satisfaction at centers (simplified):
    # sum of qhcv for each center must be >= demand
    # (some produce might not arrive if constraints are tight => spoil)
    for cobj in centers:
        cid = cobj["id"]
        dem = cobj["demand"]
        model += lpSum(qhcv[(hid,cid,vid)] for (hid,cid,vid) in y.keys() if cid==cobj["id"]) >= dem

    # 5) Spoilage variables: for each farm, total assigned to hubs must be <= farm quantity
    # spoil_f >= farm quantity - sum(qfhv) to track leftover
    for fobj in farms:
        fid = fobj["id"]
        fqty = fobj["quantity"]
        model += spoil_vars[fid] >= fqty - lpSum(qfhv[(fid,hid,vid)] for (fid,hid,vid) in x.keys() if fid==fobj["id"])

    # 6) Perish window logic (super simplified):
    # If distance farm->hub + hub->center > fobj["perish_window"], that produce spoils
    # This requires a more advanced time-based approach, but we’ll do a hack:
    # We'll impose a large penalty on any assignment that exceeds the window.
    # We can do that by forcibly setting x=0 if dist_fh + dist_hc > perish_window
    for fobj in farms:
        fid = fobj["id"]
        pw = fobj["perish_window"]
        for hobj in hubs:
            hid = hobj["id"]
            if farm_hub_dist.get((fid, hid)) is None:
                continue
            dist_fh = farm_hub_dist[(fid,hid)]
            for cobj in centers:
                cid = cobj["id"]
                if hub_center_dist.get((hid,cid)) is None:
                    continue
                dist_hc = hub_center_dist[(hid,cid)]
                total_dist = dist_fh + dist_hc
                if total_dist > pw:
                    # force that combination not to happen
                    for vobj in vehicles:
                        vid = vobj["id"]
                        if (fid,hid,vid) in x and (hid,cid,vid) in y:
                            # if x or y is set => we must not allow produce that path
                            # but we don't have direct coupling constraints x->y in a single var
                            # simplest hack: x + y <= 1 for that triple
                            # or we can just say x=0 for that route if it's guaranteed they'd use that center
                            model += x[(fid,hid,vid)] <= 0  # forcibly 0

    # Solve
    start_t = time.time()
    model.solve()
    end_t = time.time()
    status = LpStatus[model.status]
    print("Solve Status:", status)

    # Build solution
    solution = {
        "stage1_routes": [],
        "stage2_routes": [],
        "total_cost": value(model.objective),
        "total_spoilage": 0.0,
        "runtime_sec": round(end_t - start_t, 2),
        "status": status
    }

    # gather route usage
    # stage1: farm->hub
    for (fid,hid,vid), var in x.items():
        if var.varValue > 0.5:  # chosen
            distanceFH = farm_hub_dist[(fid,hid)]
            cost_multFH = farm_hub_cost_mult[(fid,hid)]
            vdata = next(vv for vv in vehicles if vv["id"] == vid)
            assigned_qty = qfhv[(fid,hid,vid)].varValue
            stage1_cost = distanceFH * vdata["cost_per_km"] * cost_multFH
            solution["stage1_routes"].append({
                "farm_id": fid,
                "hub_id": hid,
                "vehicle_id": vid,
                "assigned_qty": assigned_qty,
                "distance": distanceFH,
                "cost_multiplier": cost_multFH,
                "transport_cost": stage1_cost
            })

    # stage2: hub->center
    for (hid,cid,vid), var in y.items():
        if var.varValue > 0.5:
            distanceHC = hub_center_dist[(hid,cid)]
            cost_multHC = hub_center_cost_mult[(hid,cid)]
            vdata = next(vv for vv in vehicles if vv["id"] == vid)
            assigned_qty = qhcv[(hid,cid,vid)].varValue
            stage2_cost = distanceHC * vdata["cost_per_km"] * cost_multHC
            solution["stage2_routes"].append({
                "hub_id": hid,
                "center_id": cid,
                "vehicle_id": vid,
                "assigned_qty": assigned_qty,
                "distance": distanceHC,
                "cost_multiplier": cost_multHC,
                "transport_cost": stage2_cost
            })

    # spoilage
    total_spoil = 0
    for fid, spvar in spoil_vars.items():
        if spvar.varValue is not None:
            total_spoil += spvar.varValue
    solution["total_spoilage"] = total_spoil

    with open("novel_solution.json", "w") as f:
        json.dump(solution, f, indent=4)

    print("Novel solution saved: novel_solution.json")
    print("Objective value:", value(model.objective))
    print("Total Spoilage:", total_spoil)
    print("Runtime (sec):", round(end_t - start_t, 2))

if __name__ == "__main__":
    main()
