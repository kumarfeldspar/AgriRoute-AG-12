import streamlit as st
import pandas as pd
import json
import random
import math
import numpy as np
from copy import deepcopy

# =============================================================================
# GA Parameters
# =============================================================================
spoilage_penalty = 150
unmet_demand_penalty = 200
early_delivery_reward = 5
loading_time = 5  # constant added to travel time
POP_SIZE = 50
MAX_GENERATIONS = 50

# --------------------------------------------------------------------
# 1) Genetic Mutation Algorithm (Adapted from your "improved_GA")
# --------------------------------------------------------------------

def genetic_mutation_algorithm(
    farms, hubs, centers, vehicles,
    cost_fh, time_fh,
    cost_hc, time_hc
):
    """
    Main function that runs the Genetic Mutation Algorithm on the provided data.
    Returns: (best_individual, best_cost)
    """
    # 1) Prepare total required shipments (sum of all centers' demands).
    total_required_shipments = sum(centers[c]['demand'] for c in centers)

    def random_shipment():
        f = random.choice(list(farms.keys()))
        h = random.choice(list(hubs.keys()))
        c = random.choice(list(centers.keys()))
        v = random.choice(list(vehicles.keys()))
        return (f, h, c, v)

    def create_individual():
        """Create a random solution with length=total_required_shipments (each gene=1 unit shipped)."""
        individual = [random_shipment() for _ in range(total_required_shipments)]
        return individual

    def evaluate_individual(individual):
        """
        Compute the total cost of an individual solution.
         - Transport cost (vehicle fixed+variable plus a partial 'storage cost').
         - Spoilage penalty if route_time > farm's perishability.
         - Unmet demand penalty if total shipments to a center < its demand.
         - Vehicle capacity penalty if load > vehicle capacity.
         - Early delivery reward if delivered before center deadline.
        """
        total_cost = 0.0
        hub_usage = {h: 0 for h in hubs}
        vehicle_load = {v: 0 for v in vehicles}
        center_delivery = {c: 0 for c in centers}

        for shipment in individual:
            farm, hub, center, vehicle = shipment
            d_fh = cost_fh.get((farm, hub), 0.0)
            t_fh = time_fh.get((farm, hub), 0.0)
            d_hc = cost_hc.get((hub, center), 0.0)
            t_hc = time_hc.get((hub, center), 0.0)

            route_distance = d_fh + d_hc
            route_time = t_fh + t_hc + 2*loading_time

            veh_info = vehicles[vehicle]
            veh_cost = veh_info['fixed_cost'] + veh_info['variable_cost'] * route_distance

            # Storage cost: assume hub_time = t_hc/2
            hub_info = hubs[hub]
            storage_time = t_hc / 2.0
            storage_cost = hub_info['storage_cost_per_unit'] * storage_time

            shipment_cost = veh_cost + storage_cost

            # Spoilage penalty
            if route_time > farms[farm]['perishability_window']:
                shipment_cost += spoilage_penalty

            # Early delivery reward
            deadline = centers[center]['deadline']
            if route_time < deadline:
                shipment_cost -= early_delivery_reward * (deadline - route_time)

            total_cost += shipment_cost

            # Update usage stats
            hub_usage[hub] += 1
            vehicle_load[vehicle] += 1
            center_delivery[center] += 1

        # Add fixed hub usage cost if used
        for h in hub_usage:
            if hub_usage[h] > 0:
                total_cost += hubs[h]['fixed_usage_cost']

        # Unmet demand penalty
        for c in center_delivery:
            if center_delivery[c] < centers[c]['demand']:
                deficit = centers[c]['demand'] - center_delivery[c]
                total_cost += deficit * unmet_demand_penalty

        # Vehicle capacity penalty
        for v in vehicle_load:
            if vehicle_load[v] > vehicles[v]['capacity']:
                overload = vehicle_load[v] - vehicles[v]['capacity']
                total_cost += overload * 1000.0

        return total_cost

        # or define a separate fitness

    def fitness(ind):
        # We want to minimize cost => fitness = 1 / (cost+epsilon)
        return 1.0 / (evaluate_individual(ind) + 1e-9)

    # Selection
    def roulette_wheel_selection(pop, fits):
        total = sum(fits)
        pick = random.uniform(0, total)
        current = 0.0
        for ind, fit in zip(pop, fits):
            current += fit
            if current >= pick:
                return ind
        return pop[-1]

    def adaptive_probabilities(fit_val, avg_fit, best_fit, pc_range=(0.6, 0.9), pm_range=(0.001, 0.05)):
        pc_min, pc_max = pc_range
        pm_min, pm_max = pm_range
        ratio = (fit_val - avg_fit) / (best_fit + 1e-9)
        if fit_val >= avg_fit:
            pc = pc_min + math.cos(ratio * (math.pi/2))*(pc_max - pc_min)
            pm = pm_min
        else:
            pc = pc_max
            pm = pm_min + math.cos(ratio*(math.pi/2))*(pm_max - pm_min)
        return pc, pm

    def crossover(p1, p2, pc):
        if random.random() > pc:
            return deepcopy(p1), deepcopy(p2)
        size = len(p1)
        point1 = random.randint(0, size-2)
        point2 = random.randint(point1+1, size-1)
        c1 = p1[:point1] + p2[point1:point2] + p1[point2:]
        c2 = p2[:point1] + p1[point1:point2] + p2[point2:]
        return c1, c2

    def mutate(ind, pm):
        new_ind = deepcopy(ind)
        for i in range(len(new_ind)):
            if random.random() < pm:
                new_ind[i] = random_shipment()
        return new_ind

    # Initialize population
    population = [create_individual() for _ in range(POP_SIZE)]
    best_individual = None
    best_cost = float('inf')

    for gen in range(MAX_GENERATIONS):
        fits = [fitness(ind) for ind in population]
        costs = [evaluate_individual(ind) for ind in population]
        avg_fit = sum(fits)/len(fits)

        best_idx = np.argmin(costs)
        current_best_cost = costs[best_idx]
        current_best_fit = fits[best_idx]
        if current_best_cost < best_cost:
            best_cost = current_best_cost
            best_individual = deepcopy(population[best_idx])

        # (Optional) print progress
        if gen % 10 == 0:
            print(f"Generation {gen}: Best Cost = {best_cost:.2f}")

        # Build new population
        new_pop = []
        while len(new_pop) < POP_SIZE:
            p1 = roulette_wheel_selection(population, fits)
            p2 = roulette_wheel_selection(population, fits)
            pc, pm = adaptive_probabilities(fitness(p1), avg_fit, current_best_fit)
            c1, c2 = crossover(p1, p2, pc)
            c1 = mutate(c1, pm)
            c2 = mutate(c2, pm)
            new_pop.extend([c1, c2])

        population = new_pop[:POP_SIZE]

    return best_individual, best_cost

# --------------------------------------------------------------------
# 2) Streamlit UI
# --------------------------------------------------------------------

def main():
    st.title("Genetic Mutation Algorithm for Multi-Stage Agricultural Supply Chain")

    st.markdown("""
    This application runs a **Genetic Algorithm** that assigns shipments 
    from Farms → Hubs → Centers using different vehicles, minimizing total cost 
    (transport + storage + spoilage penalties, etc.).
    """)

    # Let user choose data input mode
    mode = st.sidebar.radio("Data Input Mode:", ["Load from JSON", "Manual Input"])

    # We'll keep data in st.session_state
    if "data" not in st.session_state:
        st.session_state["data"] = {
            "farms": {},
            "hubs": {},
            "centers": {},
            "vehicles": {},
            "cost_fh": {},
            "time_fh": {},
            "cost_hc": {},
            "time_hc": {}
        }

    if mode == "Load from JSON":
        st.subheader("Upload JSON File")
        uploaded_file = st.file_uploader("Select JSON with farms, hubs, centers, vehicles, cost/time matrices", type=["json"])
        if uploaded_file is not None:
            data_json = json.load(uploaded_file)
            # Overwrite session data
            st.session_state["data"] = data_json
            st.success("Data loaded from JSON.")
        
    else:
        st.subheader("Manual Table Input")

        # We can store each entity in a small data frame to be edited.
        # Convert st.session_state["data"]["farms"] from dict => DataFrame
        # For an example, we define columns: id, location_x, location_y, produce_quantity, perishability_window
        def dict_to_farms_df(fdict):
            rows = []
            for k,v in fdict.items():
                rows.append({
                    "id": v.get("id",k),
                    "loc_x": v["location"][0],
                    "loc_y": v["location"][1],
                    "produce_quantity": v["produce_quantity"],
                    "perishability_window": v["perishability_window"]
                })
            return pd.DataFrame(rows)

        def farms_df_to_dict(df):
            out = {}
            for _, row in df.iterrows():
                fid = int(row["id"])
                out[fid] = {
                    "id": fid,
                    "location": (float(row["loc_x"]), float(row["loc_y"])),
                    "produce_quantity": float(row["produce_quantity"]),
                    "perishability_window": float(row["perishability_window"])
                }
            return out

        def dict_to_hubs_df(hdict):
            rows = []
            for k,v in hdict.items():
                rows.append({
                    "id": v.get("id",k),
                    "loc_x": v["location"][0],
                    "loc_y": v["location"][1],
                    "capacity": v["capacity"],
                    "storage_cost_per_unit": v["storage_cost_per_unit"],
                    "fixed_usage_cost": v["fixed_usage_cost"]
                })
            return pd.DataFrame(rows)

        def hubs_df_to_dict(df):
            out = {}
            for _, row in df.iterrows():
                hid = int(row["id"])
                out[hid] = {
                    "id": hid,
                    "location": (float(row["loc_x"]), float(row["loc_y"])),
                    "capacity": float(row["capacity"]),
                    "storage_cost_per_unit": float(row["storage_cost_per_unit"]),
                    "fixed_usage_cost": float(row["fixed_usage_cost"])
                }
            return out

        def dict_to_centers_df(cdict):
            rows = []
            for k,v in cdict.items():
                rows.append({
                    "id": v.get("id",k),
                    "loc_x": v["location"][0],
                    "loc_y": v["location"][1],
                    "demand": v["demand"],
                    "deadline": v["deadline"]
                })
            return pd.DataFrame(rows)

        def centers_df_to_dict(df):
            out = {}
            for _, row in df.iterrows():
                cid = int(row["id"])
                out[cid] = {
                    "id": cid,
                    "location": (float(row["loc_x"]), float(row["loc_y"])),
                    "demand": float(row["demand"]),
                    "deadline": float(row["deadline"])
                }
            return out

        def dict_to_vehicles_df(vdict):
            rows = []
            for k,v in vdict.items():
                rows.append({
                    "id": v.get("id", k),
                    "type": v["type"],
                    "capacity": v["capacity"],
                    "fixed_cost": v["fixed_cost"],
                    "variable_cost": v["variable_cost"]
                })
            return pd.DataFrame(rows)

        def vehicles_df_to_dict(df):
            out = {}
            for _, row in df.iterrows():
                vid = int(row["id"])
                out[vid] = {
                    "id": vid,
                    "type": str(row["type"]),
                    "capacity": float(row["capacity"]),
                    "fixed_cost": float(row["fixed_cost"]),
                    "variable_cost": float(row["variable_cost"])
                }
            return out

        # For cost/time dict => DataFrame. The keys are like "(farm_id, hub_id)", value is distance or time
        def dict_to_cost_df(ddict):
            rows = []
            for k,v in ddict.items():
                rows.append({
                    "edge_key": str(k),  # e.g. "(0,1)"
                    "value": v
                })
            return pd.DataFrame(rows)

        def cost_df_to_dict(df):
            out = {}
            for _, row in df.iterrows():
                # parse the edge_key => tuple
                ek = row["edge_key"].strip()
                # e.g. "(0,1)"
                ek_tuple = eval(ek)  # unsafe but quick for demonstration
                out[ek_tuple] = float(row["value"])
            return out

        # Show data editors
        st.markdown("**Farms**")
        farms_df = dict_to_farms_df(st.session_state["data"].get("farms", {}))
        farms_edited = st.data_editor(farms_df, num_rows="dynamic", use_container_width=True)
        
        st.markdown("**Hubs**")
        hubs_df = dict_to_hubs_df(st.session_state["data"].get("hubs", {}))
        hubs_edited = st.data_editor(hubs_df, num_rows="dynamic", use_container_width=True)

        st.markdown("**Centers**")
        centers_df = dict_to_centers_df(st.session_state["data"].get("centers", {}))
        centers_edited = st.data_editor(centers_df, num_rows="dynamic", use_container_width=True)

        st.markdown("**Vehicles**")
        vehicles_df = dict_to_vehicles_df(st.session_state["data"].get("vehicles", {}))
        vehicles_edited = st.data_editor(vehicles_df, num_rows="dynamic", use_container_width=True)

        st.markdown("**Cost Farm→Hub (cost_fh)**")
        costfh_df = dict_to_cost_df(st.session_state["data"].get("cost_fh", {}))
        costfh_edited = st.data_editor(costfh_df, num_rows="dynamic", use_container_width=True)

        st.markdown("**Time Farm→Hub (time_fh)**")
        timefh_df = dict_to_cost_df(st.session_state["data"].get("time_fh", {}))
        timefh_edited = st.data_editor(timefh_df, num_rows="dynamic", use_container_width=True)

        st.markdown("**Cost Hub→Center (cost_hc)**")
        costhc_df = dict_to_cost_df(st.session_state["data"].get("cost_hc", {}))
        costhc_edited = st.data_editor(costhc_df, num_rows="dynamic", use_container_width=True)

        st.markdown("**Time Hub→Center (time_hc)**")
        timehc_df = dict_to_cost_df(st.session_state["data"].get("time_hc", {}))
        timehc_edited = st.data_editor(timehc_df, num_rows="dynamic", use_container_width=True)

        if st.button("Save Manual Data"):
            # Convert back to dict
            st.session_state["data"]["farms"] = farms_df_to_dict(farms_edited)
            st.session_state["data"]["hubs"] = hubs_df_to_dict(hubs_edited)
            st.session_state["data"]["centers"] = centers_df_to_dict(centers_edited)
            st.session_state["data"]["vehicles"] = vehicles_df_to_dict(vehicles_edited)
            st.session_state["data"]["cost_fh"] = cost_df_to_dict(costfh_edited)
            st.session_state["data"]["time_fh"] = cost_df_to_dict(timefh_edited)
            st.session_state["data"]["cost_hc"] = cost_df_to_dict(costhc_edited)
            st.session_state["data"]["time_hc"] = cost_df_to_dict(timehc_edited)
            st.success("Manual data saved to session.")

    st.subheader("Run Genetic Mutation Algorithm")
    if st.button("Run Algorithm"):
        data = st.session_state["data"]
        if not data["farms"] or not data["hubs"] or not data["centers"] or not data["vehicles"]:
            st.error("Farms, Hubs, Centers, Vehicles data are required.")
        else:
            best_solution, best_cost = genetic_mutation_algorithm(
                farms=data["farms"],
                hubs=data["hubs"],
                centers=data["centers"],
                vehicles=data["vehicles"],
                cost_fh=data["cost_fh"],
                time_fh=data["time_fh"],
                cost_hc=data["cost_hc"],
                time_hc=data["time_hc"]
            )
            st.success(f"Algorithm finished. Best Cost found: {best_cost:.2f}")

            # Display the best solution
            st.markdown("### Best Solution (Shipments)")
            shipments_df = []
            for i, shp in enumerate(best_solution):
                farm, hub, center, vehicle = shp
                # Calculate partial cost for that shipment
                d_fh = data["cost_fh"].get((farm, hub), 0)
                d_hc = data["cost_hc"].get((hub, center), 0)
                route_dist = d_fh + d_hc
                veh_info = data["vehicles"][vehicle]
                route_cost = veh_info["fixed_cost"] + veh_info["variable_cost"]*route_dist
                total_time = data["time_fh"].get((farm,hub),0) + data["time_hc"].get((hub,center),0) + 2*loading_time
                shipments_df.append({
                    "Shipment#": i+1,
                    "Farm": farm,
                    "Hub": hub,
                    "Center": center,
                    "Vehicle": vehicle,
                    "RouteDistance": route_dist,
                    "ShipmentCost(approx)": round(route_cost,2),
                    "TotalTime": round(total_time,2)
                })
            st.dataframe(pd.DataFrame(shipments_df))

            # Aggregated stats
            farm_ship = {f:0 for f in data["farms"]}
            hub_ship = {h:0 for h in data["hubs"]}
            ctr_ship = {c:0 for c in data["centers"]}
            veh_routes = {v: [] for v in data["vehicles"]}
            for shp in best_solution:
                f, h, c, v = shp
                farm_ship[f] += 1
                hub_ship[h] += 1
                ctr_ship[c] += 1
                veh_routes[v].append(shp)

            st.markdown("### Aggregated Shipments:")
            st.write("**Farms →**", farm_ship)
            st.write("**Hubs →**", hub_ship)
            st.write("**Centers →**", ctr_ship)

            st.markdown("### Detailed Routes per Vehicle:")
            for v in veh_routes:
                st.write(f"**Vehicle {v}** (type={data['vehicles'][v]['type']}, capacity={data['vehicles'][v]['capacity']}):")
                table_v = []
                for idx, route in enumerate(veh_routes[v]):
                    f,h,c,_ = route
                    route_dist = data["cost_fh"].get((f,h),0) + data["cost_hc"].get((h,c),0)
                    route_time = data["time_fh"].get((f,h),0) + data["time_hc"].get((h,c),0) + 2*loading_time
                    rcost = data["vehicles"][v]["fixed_cost"] + data["vehicles"][v]["variable_cost"]*route_dist
                    table_v.append({
                        "Shipment#": idx+1,
                        "Farm": f, "Hub": h, "Center": c,
                        "Distance": round(route_dist,2),
                        "Time": round(route_time,2),
                        "Cost": round(rcost,2)
                    })
                if table_v:
                    st.dataframe(pd.DataFrame(table_v))
                else:
                    st.write("No shipments for this vehicle.")

if __name__ == "__main__":
    main()
