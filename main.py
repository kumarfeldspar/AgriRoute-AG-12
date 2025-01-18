#!/usr/bin/env python
"""
main.py
-------
Integrated Streamlit Dashboard for Agro-Route, including analytics/visualization.

Usage:
    streamlit run main.py
"""

import streamlit as st
import pandas as pd
import altair as alt
import math
import matplotlib.pyplot as plt
import plotly.express as px
import json
import os

# Folium-based imports
import folium
from streamlit_folium import st_folium

from data_input import (
    simulate_farms, simulate_storage_hubs,
    simulate_distribution_centers, simulate_vehicles
)
from distance_azure import build_distance_matrix
from greedy_kmeans import run_greedy_kmeans
from linear_programming import run_linear_program
from ga_mutation import run_genetic_algorithm

# -----------------------------------------------------------
# Utility functions to persist or load map data
# -----------------------------------------------------------
def load_map_data():
    """
    Loads map data (farms, hubs, centers, vehicles, dist_dict)
    from map_data.json, if it exists, into st.session_state.

    We convert dist_dict string keys back into tuple keys:
        "farm_0_hub_0" -> ("farm", 0, "hub", 0)
    """
    if os.path.exists("map_data.json"):
        try:
            with open("map_data.json", "r") as f:
                data = json.load(f)
            # Load direct lists for farms, hubs, centers, vehicles
            for k in ["farms", "hubs", "centers", "vehicles"]:
                if k in data:
                    st.session_state[k] = data[k]
            # Re-convert dist_dict string keys to tuple keys
            dist_loaded = {}
            if "dist_dict" in data:
                for k_str, val in data["dist_dict"].items():
                    parts = k_str.split("_")
                    if len(parts) == 4:
                        t1, id1_str, t2, id2_str = parts
                        dist_loaded[(t1, int(id1_str), t2, int(id2_str))] = val
            st.session_state["dist_dict"] = dist_loaded
        except Exception as e:
            st.error(f"Error loading map_data.json: {e}")

def save_map_data():
    """
    Saves map data (farms, hubs, centers, vehicles, dist_dict)
    from st.session_state into map_data.json.

    We convert tuple keys in dist_dict to strings:
        ("farm", 0, "hub", 0) -> "farm_0_hub_0"
    """
    data = {}
    # Store direct lists
    for k in ["farms", "hubs", "centers", "vehicles"]:
        data[k] = st.session_state.get(k, [])

    # Convert dist_dict's tuple keys to string
    dist_data = st.session_state.get("dist_dict", {})
    dist_str_dict = {}
    for tuple_key, val in dist_data.items():
        if len(tuple_key) == 4:
            t1, id1, t2, id2 = tuple_key
            key_str = f"{t1}_{id1}_{t2}_{id2}"
            dist_str_dict[key_str] = val
    data["dist_dict"] = dist_str_dict

    # Write out to JSON
    try:
        with open("map_data.json", "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.error(f"Error saving map_data.json: {e}")


# -----------------------------------------------------------
# Function to build a Folium map of all farms/hubs/centers 
# and optionally show solution routes from Greedy, LP, and GA.
# -----------------------------------------------------------
def create_solution_map():
    """
    Creates a Folium map displaying:
      - All farms, hubs, and centers as markers.
      - (Optional) polylines for each solution in session_state:
         * Greedy solution (clusters) => Purple lines
         * LP solution (detailed_routes) => Blue/Orange lines
         * GA solution (detailed_routes) => Green/Red lines
    """
    # Prepare the base Folium map
    initial_coords = [28.7, 77.1]
    fol_map = folium.Map(location=initial_coords, zoom_start=8)

    # Pull data from session_state
    farms = st.session_state.get("farms", [])
    hubs = st.session_state.get("hubs", [])
    centers = st.session_state.get("centers", [])

    # Convert them to dict keyed by ID for quick lookup
    farm_dict = {f['id']: f for f in farms}
    hub_dict = {h['id']: h for h in hubs}
    center_dict = {c['id']: c for c in centers}

    # ---------------------------
    # 1) Add markers for Farms, Hubs, Centers
    # ---------------------------
    # Farms (green icons)
    for f in farms:
        lat, lng = f["location"]
        popup_text = (
            f"<b>Farm #{f['id']}</b><br>"
            f"Produce Qty: {f['produce_quantity']}<br>"
            f"Perishability: {f['perishability_window']} days"
        )
        folium.Marker(
            location=[lat, lng],
            popup=popup_text,
            tooltip=f"Farm #{f['id']}",
            icon=folium.Icon(color="green", icon="leaf", prefix="fa")
        ).add_to(fol_map)

    # Hubs (blue icons)
    for h in hubs:
        lat, lng = h["location"]
        popup_text = (
            f"<b>Hub #{h['id']}</b><br>"
            f"Capacity: {h['capacity']}<br>"
            f"Storage Cost/Unit: {h['storage_cost_per_unit']}<br>"
            f"Fixed Usage Cost: {h['fixed_usage_cost']}"
        )
        folium.Marker(
            location=[lat, lng],
            popup=popup_text,
            tooltip=f"Hub #{h['id']}",
            icon=folium.Icon(color="blue", icon="warehouse", prefix="fa")
        ).add_to(fol_map)

    # Centers (red icons)
    for c in centers:
        lat, lng = c["location"]
        popup_text = (
            f"<b>Center #{c['id']}</b><br>"
            f"Demand: {c['demand']}<br>"
            f"Deadline: {c['deadline']} days"
        )
        folium.Marker(
            location=[lat, lng],
            popup=popup_text,
            tooltip=f"Center #{c['id']}",
            icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa")
        ).add_to(fol_map)

    # ---------------------------
    # 2) Add routes from solutions if available
    # ---------------------------
    # a) Greedy solution => "clusters" approach
    #    We'll draw farm->hub in purple. If a center_id is found, hub->center in purple.
    greedy_sol = st.session_state.get("greedy_solution", {})
    if greedy_sol:
        clusters = greedy_sol.get("clusters", [])
        for cl in clusters:
            hub_id = cl.get("hub_id")
            if hub_id not in hub_dict:
                continue
            hub_loc = hub_dict[hub_id]["location"]
            # farm->hub
            for cfarm in cl.get("cluster_farms", []):
                fid = cfarm.get("farm_id")
                if fid in farm_dict:
                    f_loc = farm_dict[fid]["location"]
                    farm_hub_coords = [(f_loc[0], f_loc[1]), (hub_loc[0], hub_loc[1])]
                    folium.PolyLine(
                        locations=farm_hub_coords,
                        color="purple",
                        weight=3,
                        tooltip="Greedy Route",
                        popup=f"<b>Greedy</b><br>Farm #{fid} → Hub #{hub_id}"
                    ).add_to(fol_map)
            # hub->center if known
            center_id = cl.get("center_id", None)
            if center_id in center_dict:
                center_loc = center_dict[center_id]["location"]
                hub_center_coords = [(hub_loc[0], hub_loc[1]), (center_loc[0], center_loc[1])]
                folium.PolyLine(
                    locations=hub_center_coords,
                    color="purple",
                    weight=3,
                    tooltip="Greedy Route",
                    popup=f"<b>Greedy</b><br>Hub #{hub_id} → Center #{center_id}"
                ).add_to(fol_map)

    # b) LP solution => "detailed_routes"
    #    We'll do blue if not spoiled, orange if spoiled (if "spoiled" is present).
    lp_sol = st.session_state.get("lp_solution", {})
    if lp_sol:
        lp_routes = lp_sol.get("detailed_routes", [])
        for route in lp_routes:
            f_id = route["farm_id"]
            h_id = route["hub_id"]
            c_id = route["center_id"]
            # farm->hub
            if f_id in farm_dict and h_id in hub_dict:
                f_lat, f_lng = farm_dict[f_id]["location"]
                h_lat, h_lng = hub_dict[h_id]["location"]
                color_fh = "blue"
                if route.get("spoiled") is True:
                    color_fh = "orange"
                folium.PolyLine(
                    locations=[(f_lat, f_lng), (h_lat, h_lng)],
                    color=color_fh,
                    weight=4,
                    tooltip="LP Route (Farm→Hub)",
                    popup=(
                        f"<b>LP Route</b><br>"
                        f"Farm #{f_id} → Hub #{h_id}<br>"
                        f"Distance: {route.get('dist_farm_hub','N/A')} km<br>"
                        f"Spoiled: {route.get('spoiled','N/A')}"
                    )
                ).add_to(fol_map)
            # hub->center
            if h_id in hub_dict and c_id in center_dict:
                h_lat, h_lng = hub_dict[h_id]["location"]
                c_lat, c_lng = center_dict[c_id]["location"]
                color_hc = "blue"
                if route.get("spoiled") is True:
                    color_hc = "orange"
                folium.PolyLine(
                    locations=[(h_lat, h_lng), (c_lat, c_lng)],
                    color=color_hc,
                    weight=4,
                    tooltip="LP Route (Hub→Center)",
                    popup=(
                        f"<b>LP Route</b><br>"
                        f"Hub #{h_id} → Center #{c_id}<br>"
                        f"Distance: {route.get('dist_hub_center','N/A')} km<br>"
                        f"Spoiled: {route.get('spoiled','N/A')}"
                    )
                ).add_to(fol_map)

    # c) GA solution => "detailed_routes"
    #    We'll do green if not spoiled, red if spoiled.
    ga_sol = st.session_state.get("ga_solution", {})
    if ga_sol:
        ga_routes = ga_sol.get("detailed_routes", [])
        for route in ga_routes:
            f_id = route["farm_id"]
            h_id = route["hub_id"]
            c_id = route["center_id"]
            # farm->hub
            if f_id in farm_dict and h_id in hub_dict:
                f_lat, f_lng = farm_dict[f_id]["location"]
                h_lat, h_lng = hub_dict[h_id]["location"]
                color_fh = "green" if not route.get("spoiled") else "red"
                folium.PolyLine(
                    locations=[(f_lat, f_lng), (h_lat, h_lng)],
                    color=color_fh,
                    weight=4,
                    tooltip="GA Route (Farm→Hub)",
                    popup=(
                        f"<b>GA Route</b><br>"
                        f"Farm #{f_id} → Hub #{h_id}<br>"
                        f"Distance: {route.get('dist_farm_hub','N/A')} km<br>"
                        f"Route Cost: {route.get('route_cost','N/A')}<br>"
                        f"Spoiled: {route.get('spoiled','N/A')}"
                    )
                ).add_to(fol_map)
            # hub->center
            if h_id in hub_dict and c_id in center_dict:
                h_lat, h_lng = hub_dict[h_id]["location"]
                c_lat, c_lng = center_dict[c_id]["location"]
                color_hc = "green" if not route.get("spoiled") else "red"
                folium.PolyLine(
                    locations=[(h_lat, h_lng), (c_lat, c_lng)],
                    color=color_hc,
                    weight=4,
                    tooltip="GA Route (Hub→Center)",
                    popup=(
                        f"<b>GA Route</b><br>"
                        f"Hub #{h_id} → Center #{c_id}<br>"
                        f"Distance: {route.get('dist_hub_center','N/A')} km<br>"
                        f"Route Cost: {route.get('route_cost','N/A')}<br>"
                        f"Spoiled: {route.get('spoiled','N/A')}"
                    )
                ).add_to(fol_map)

    return fol_map


# -----------------------------------------------------------
# Analytics/Comparison Section
# -----------------------------------------------------------
def show_comparisons():
    """
    Create a comparison of LP, Greedy, and GA solutions stored in st.session_state.
    Plots various charts to analyze total cost/spoilage cost.
    """
    lp_data = st.session_state.get("lp_solution", {})
    greedy_data = st.session_state.get("greedy_solution", {})
    ga_data = st.session_state.get("ga_solution", {})

    methods = ['Linear Programming', 'Greedy-KMeans', 'Genetic Algorithm']
    lp_total_cost = lp_data.get('best_cost', 0)
    lp_spoilage_cost = lp_data.get('total_spoilage_cost', 0)

    gr_total_cost = greedy_data.get('best_cost', 0)
    gr_spoilage_cost = greedy_data.get('total_spoilage_cost', 0)

    ga_total_cost = ga_data.get('best_cost', 0)
    ga_spoilage_cost = ga_data.get('total_spoilage_cost', 0)

    data = {
        'Method': methods,
        'Total Cost': [
            lp_total_cost,
            gr_total_cost,
            ga_total_cost
        ],
        'Total Spoilage Cost': [
            lp_spoilage_cost,
            gr_spoilage_cost,
            ga_spoilage_cost
        ]
    }
    df = pd.DataFrame(data)

    if df['Total Cost'].sum() == 0 and df['Total Spoilage Cost'].sum() == 0:
        st.warning("No valid solutions found in session. Generate or load solutions first.")
        return

    st.title("Optimization Methods Analytics")
    st.write("Analyze and compare the performance of optimization methods based on total cost and spoilage cost.")

    # Sidebar filters
    st.sidebar.header("Filter Methods")
    methods_to_display = st.sidebar.multiselect(
        "Select Methods to Display",
        methods,
        default=methods
    )
    filtered_df = df[df['Method'].isin(methods_to_display)]

    st.subheader("Comparison Data")
    st.dataframe(filtered_df)

    st.subheader("Cost Comparisons")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Total Cost")
        fig, ax = plt.subplots()
        ax.bar(filtered_df['Method'], filtered_df['Total Cost'], color=['blue', 'green', 'orange'])
        ax.set_xlabel("Method")
        ax.set_ylabel("Total Cost")
        ax.set_title("Total Cost by Method")
        st.pyplot(fig)

    with col2:
        st.write("### Total Spoilage Cost")
        fig, ax = plt.subplots()
        ax.bar(filtered_df['Method'], filtered_df['Total Spoilage Cost'], color=['blue', 'green', 'orange'])
        ax.set_xlabel("Method")
        ax.set_ylabel("Total Spoilage Cost")
        ax.set_title("Total Spoilage Cost by Method")
        st.pyplot(fig)

    st.subheader("Proportional Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Proportion of Total Cost")
        fig_pie_cost = px.pie(
            filtered_df,
            values="Total Cost",
            names="Method",
            title="Total Cost Proportions"
        )
        st.plotly_chart(fig_pie_cost)

    with col2:
        st.write("### Proportion of Spoilage Cost")
        fig_pie_spoilage = px.pie(
            filtered_df,
            values="Total Spoilage Cost",
            names="Method",
            title="Total Spoilage Cost Proportions"
        )
        st.plotly_chart(fig_pie_spoilage)

    st.subheader("Trends Across Methods")
    df_melt = filtered_df.melt(id_vars=["Method"], value_vars=["Total Cost", "Total Spoilage Cost"])
    fig_line = px.line(
        df_melt,
        x="Method", y="value", color="variable",
        title="Trends: Total Cost vs Total Spoilage Cost",
        labels={"value": "Cost", "variable": "Cost Type"}
    )
    st.plotly_chart(fig_line)

    st.subheader("Cost Relationships")
    fig_scatter = px.scatter(
        filtered_df,
        x="Total Cost",
        y="Total Spoilage Cost",
        color="Method",
        size="Total Spoilage Cost",
        title="Total Cost vs Total Spoilage Cost",
        labels={"Total Cost": "Total Cost", "Total Spoilage Cost": "Total Spoilage Cost"}
    )
    st.plotly_chart(fig_scatter)

    st.subheader("Insights")
    if len(filtered_df) > 0:
        min_cost_method = filtered_df.loc[filtered_df['Total Cost'].idxmin(), 'Method']
        min_spoilage_method = filtered_df.loc[filtered_df['Total Spoilage Cost'].idxmin(), 'Method']
        max_cost_method = filtered_df.loc[filtered_df['Total Cost'].idxmax(), 'Method']
        max_spoilage_method = filtered_df.loc[filtered_df['Total Spoilage Cost'].idxmax(), 'Method']

        st.write(f"### Lowest Total Cost:")
        st.write(f"- *Method:* {min_cost_method}")
        st.write(f"- *Cost:* {filtered_df['Total Cost'].min()}")

        st.write(f"### Lowest Total Spoilage Cost:")
        st.write(f"- *Method:* {min_spoilage_method}")
        st.write(f"- *Spoilage Cost:* {filtered_df['Total Spoilage Cost'].min()}")

        st.write(f"### Highest Total Cost:")
        st.write(f"- *Method:* {max_cost_method}")
        st.write(f"- *Cost:* {filtered_df['Total Cost'].max()}")

        st.write(f"### Highest Total Spoilage Cost:")
        st.write(f"- *Method:* {max_spoilage_method}")
        st.write(f"- *Spoilage Cost:* {filtered_df['Total Spoilage Cost'].max()}")
    else:
        st.info("No methods selected or no data to display insights.")


# -----------------------------------------------------------
# Main Dashboard
# -----------------------------------------------------------
def main():
    st.title("Agro-Route: Optimized Agricultural Supply Chain System")

    # Attempt to load existing map_data into session_state if not set
    if "farms" not in st.session_state:
        load_map_data()

    # Initialize session state objects if they are not set
    for key in ["farms", "hubs", "centers", "vehicles", "dist_dict"]:
        if key not in st.session_state:
            st.session_state[key] = [] if key != "dist_dict" else {}

    # Also initialize a "show_map" state so the map persists
    if "show_map" not in st.session_state:
        st.session_state["show_map"] = False

    st.sidebar.title("1) Data Input Options")
    input_mode = st.sidebar.radio("Choose Input Mode:", ["Manual", "Simulate"])

    # ----------------------------
    # MANUAL INPUT MODE
    # ----------------------------
    if input_mode == "Manual":
        st.sidebar.subheader("Enter Numbers of Entities")
        num_farms = st.sidebar.number_input("Number of Farms", min_value=1, max_value=50, value=3)
        num_hubs = st.sidebar.number_input("Number of Hubs", min_value=1, max_value=20, value=2)
        num_centers = st.sidebar.number_input("Number of Centers", min_value=1, max_value=20, value=2)
        num_vehicles = st.sidebar.number_input("Number of Vehicles", min_value=1, max_value=10, value=3)

        if st.sidebar.button("Create Manual Forms"):
            st.session_state["farms"] = [
                {
                    "id": i,
                    "location": [28.7 + i * 0.01, 77.1 + i * 0.01],
                    "produce_quantity": 500,
                    "perishability_window": 3
                }
                for i in range(num_farms)
            ]
            st.session_state["hubs"] = [
                {
                    "id": i,
                    "location": [28.6 + i * 0.01, 77.0 + i * 0.01],
                    "capacity": 2000,
                    "storage_cost_per_unit": 1.0,
                    "fixed_usage_cost": 1000
                }
                for i in range(num_hubs)
            ]
            st.session_state["centers"] = [
                {
                    "id": i,
                    "location": [28.9 + i * 0.01, 77.2 + i * 0.01],
                    "demand": 800,
                    "deadline": 4
                }
                for i in range(num_centers)
            ]
            st.session_state["vehicles"] = [
                {
                    "id": i,
                    "type": "small" if i < 2 else "large",
                    "capacity": 1000 if i < 2 else 3000,
                    "fixed_cost": 200 if i < 2 else 350,
                    "variable_cost_per_distance": 0.8 if i < 2 else 1.0
                }
                for i in range(num_vehicles)
            ]
            st.session_state["dist_dict"] = {}
            st.success("Default manual forms created!")
            save_map_data()

        st.header("Manual Edit of Data")

        # --- Farms Form ---
        with st.form("farms_form"):
            st.subheader("Farms")
            new_farms = []
            for i, f in enumerate(st.session_state["farms"]):
                st.markdown(f"**Farm #{i}**")
                fid = st.number_input(f"Farm ID", value=f["id"], key=f"farm_id_{i}")
                lat = st.number_input(f"Latitude", value=float(f["location"][0]), key=f"farm_lat_{i}")
                lon = st.number_input(f"Longitude", value=float(f["location"][1]), key=f"farm_lon_{i}")
                qty = st.number_input(f"Produce Quantity", value=int(f["produce_quantity"]), key=f"farm_qty_{i}")
                perish = st.number_input(f"Perishability Window", value=int(f["perishability_window"]), key=f"farm_pw_{i}")
                new_farms.append({
                    "id": fid,
                    "location": [lat, lon],
                    "produce_quantity": qty,
                    "perishability_window": perish
                })
                st.markdown("---")
            if st.form_submit_button("Save Farms"):
                st.session_state["farms"] = new_farms
                st.success("Farms updated!")
                save_map_data()

        # --- Hubs Form ---
        with st.form("hubs_form"):
            st.subheader("Storage Hubs")
            new_hubs = []
            for i, h in enumerate(st.session_state["hubs"]):
                st.markdown(f"**Hub #{i}**")
                hid = st.number_input(f"Hub ID", value=h["id"], key=f"hub_id_{i}")
                lat = st.number_input(f"Latitude", value=float(h["location"][0]), key=f"hub_lat_{i}")
                lon = st.number_input(f"Longitude", value=float(h["location"][1]), key=f"hub_lon_{i}")
                cap = st.number_input(f"Capacity", value=int(h["capacity"]), key=f"hub_cap_{i}")
                scost = st.number_input(f"Storage Cost per Unit", value=float(h["storage_cost_per_unit"]), key=f"hub_scost_{i}")
                fcost = st.number_input(f"Fixed Usage Cost", value=float(h["fixed_usage_cost"]), key=f"hub_fcost_{i}")
                new_hubs.append({
                    "id": hid,
                    "location": [lat, lon],
                    "capacity": cap,
                    "storage_cost_per_unit": scost,
                    "fixed_usage_cost": fcost
                })
                st.markdown("---")
            if st.form_submit_button("Save Hubs"):
                st.session_state["hubs"] = new_hubs
                st.success("Hubs updated!")
                save_map_data()

        # --- Centers Form ---
        with st.form("centers_form"):
            st.subheader("Distribution Centers")
            new_centers = []
            for i, c in enumerate(st.session_state["centers"]):
                st.markdown(f"**Center #{i}**")
                cid = st.number_input(f"Center ID", value=c["id"], key=f"cen_id_{i}")
                lat = st.number_input(f"Latitude", value=float(c["location"][0]), key=f"cen_lat_{i}")
                lon = st.number_input(f"Longitude", value=float(c["location"][1]), key=f"cen_lon_{i}")
                dem = st.number_input(f"Demand", value=int(c["demand"]), key=f"cen_dem_{i}")
                deadline = st.number_input(f"Deadline", value=int(c["deadline"]), key=f"cen_deadline_{i}")
                new_centers.append({
                    "id": cid,
                    "location": [lat, lon],
                    "demand": dem,
                    "deadline": deadline
                })
                st.markdown("---")
            if st.form_submit_button("Save Centers"):
                st.session_state["centers"] = new_centers
                st.success("Centers updated!")
                save_map_data()

        # --- Vehicles Form ---
        with st.form("vehicles_form"):
            st.subheader("Vehicles")
            new_vehicles = []
            for i, v in enumerate(st.session_state["vehicles"]):
                st.markdown(f"**Vehicle #{i}**")
                vid = st.number_input(f"Vehicle ID", value=v["id"], key=f"veh_id_{i}")
                vtype = st.selectbox(f"Type", options=["small", "large"],
                                     index=(0 if v["type"] == "small" else 1),
                                     key=f"veh_type_{i}")
                cap = st.number_input(f"Capacity", value=int(v["capacity"]), key=f"veh_cap_{i}")
                fcost = st.number_input(f"Fixed Cost", value=float(v["fixed_cost"]), key=f"veh_fcost_{i}")
                varcost = st.number_input(f"Variable Cost Per Distance", value=float(v["variable_cost_per_distance"]), key=f"veh_varcost_{i}")
                new_vehicles.append({
                    "id": vid,
                    "type": vtype,
                    "capacity": cap,
                    "fixed_cost": fcost,
                    "variable_cost_per_distance": varcost
                })
                st.markdown("---")
            if st.form_submit_button("Save Vehicles"):
                st.session_state["vehicles"] = new_vehicles
                st.success("Vehicles updated!")
                save_map_data()

    # ----------------------------
    # SIMULATE MODE
    # ----------------------------
    elif input_mode == "Simulate":
        st.sidebar.subheader("Simulation Setup")
        num_farms_s = st.sidebar.slider("Num Farms", 1, 50, 5)
        num_hubs_s = st.sidebar.slider("Num Hubs", 1, 20, 3)
        num_centers_s = st.sidebar.slider("Num Centers", 1, 20, 2)
        small_veh = st.sidebar.slider("Small Vehicles", 1, 5, 2)
        large_veh = st.sidebar.slider("Large Vehicles", 0, 5, 1)

        if st.sidebar.button("Simulate Data"):
            st.session_state["farms"] = simulate_farms(num_farms_s, seed=42)
            st.session_state["hubs"] = simulate_storage_hubs(num_hubs_s, seed=100)
            st.session_state["centers"] = simulate_distribution_centers(num_centers_s, seed=200)
            st.session_state["vehicles"] = simulate_vehicles(small_veh, large_veh, seed=300)
            st.session_state["dist_dict"] = {}
            st.success("Simulated data created!")
            save_map_data()

    # ----------------------------
    # Display Current Data
    # ----------------------------
    st.subheader("2) Current Data")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Farms**")
        st.dataframe(pd.DataFrame(st.session_state["farms"]))
        st.write("**Storage Hubs**")
        st.dataframe(pd.DataFrame(st.session_state["hubs"]))
    with col2:
        st.write("**Distribution Centers**")
        st.dataframe(pd.DataFrame(st.session_state["centers"]))
        st.write("**Vehicles**")
        st.dataframe(pd.DataFrame(st.session_state["vehicles"]))

    # ----------------------------
    # Build Distance Matrix
    # ----------------------------
    st.subheader("3) Build Distance Matrix")
    use_azure = st.checkbox("Use Azure Maps?")
    if st.button("Generate Distances"):
        dist_d = build_distance_matrix(
            st.session_state["farms"],
            st.session_state["hubs"],
            st.session_state["centers"],
            use_azure=use_azure
        )
        st.session_state["dist_dict"] = dist_d
        st.success("Distance matrix generated.")
        save_map_data()

    if st.session_state["dist_dict"]:
        st.write("Distance Matrix (sample):")
        sample_items = list(st.session_state["dist_dict"].items())[:10]
        st.write(sample_items)

    # ----------------------------
    # Run Optimization / Benchmarks
    # ----------------------------
    st.subheader("4) Run Benchmarks / Optimization")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Greedy K-Means"):
            sol = run_greedy_kmeans(
                st.session_state["farms"],
                st.session_state["hubs"],
                st.session_state["centers"],
                st.session_state["vehicles"],
                st.session_state["dist_dict"],
                output_file="greedy_solution.json"
            )
            st.session_state["greedy_solution"] = sol
            st.success("Greedy K-Means Completed!")
    with colB:
        if st.button("Linear Programming"):
            sol = run_linear_program(
                st.session_state["farms"],
                st.session_state["hubs"],
                st.session_state["centers"],
                st.session_state["vehicles"],
                st.session_state["dist_dict"],
                output_file="lp_solution.json"
            )
            st.session_state["lp_solution"] = sol
            st.success("LP Completed!")
    with colC:
        if st.button("Genetic Algorithm"):
            sol = run_genetic_algorithm(
                st.session_state["farms"],
                st.session_state["hubs"],
                st.session_state["centers"],
                st.session_state["vehicles"],
                st.session_state["dist_dict"],
                pop_size=50,
                max_generations=50,
                output_file="ga_solution.json"
            )
            st.session_state["ga_solution"] = sol
            st.success("Genetic Algorithm Completed!")

    # ----------------------------
    # Display Solutions JSON (Optional)
    # ----------------------------
    st.subheader("5) Solutions JSON (Optional)")
    exp = st.expander("Greedy Solution")
    with exp:
        st.json(st.session_state.get("greedy_solution", {}))

    exp2 = st.expander("LP Solution")
    with exp2:
        st.json(st.session_state.get("lp_solution", {}))

    exp3 = st.expander("GA Solution")
    with exp3:
        st.json(st.session_state.get("ga_solution", {}))

    # ----------------------------
    # 6) Folium Map Visualization
    # ----------------------------
    st.subheader("6) Map Visualization")
    st.write("Below is an interactive Folium map showing farms, hubs, centers, and solution routes. "
             "Press **Refresh Map** to load or update routes from your solutions.")

    if st.button("Refresh Map"):
        st.session_state["show_map"] = True  # Turn on map display

    # If show_map is True, create and display the Folium map
    if st.session_state["show_map"]:
        fol_map = create_solution_map()
        st_folium(fol_map, width=1000, height=600)

    # # ----------------------------
    # # 7) Timeline (Concept)
    # # ----------------------------
    # st.subheader("7) Timeline (Concept)")
    # timeline_data = []
    # lp_sol = st.session_state.get("lp_solution")
    # gr_sol = st.session_state.get("greedy_solution")
    # ga_sol = st.session_state.get("ga_solution")

    # # Example for LP
    # if lp_sol and "detailed_routes" in lp_sol:
    #     for route in lp_sol["detailed_routes"]:
    #         fid = route["farm_id"]
    #         hid = route["hub_id"]
    #         cid = route["center_id"]
    #         timeline_data.append({
    #             "entity": f"Farm{fid}->Hub{hid}",
    #             "start": 0,
    #             "end": 1,
    #             "stage": "Farm->Hub"
    #         })
    #         timeline_data.append({
    #             "entity": f"Hub{hid}->Center{cid}",
    #             "start": 1,
    #             "end": 2,
    #             "stage": "Hub->Center"
    #         })
    # # Example for Greedy
    # elif gr_sol and "clusters" in gr_sol:
    #     for cl in gr_sol.get("clusters", []):
    #         hid = cl.get("hub_id")
    #         if hid is not None:
    #             for cf in cl.get("cluster_farms", []):
    #                 timeline_data.append({
    #                     "entity": f"Farm{cf['farm_id']}->Hub{hid}",
    #                     "start": 0,
    #                     "end": 1,
    #                     "stage": "Farm->Hub"
    #                 })
    #             cid = cl.get("center_id")
    #             if cid is not None:
    #                 timeline_data.append({
    #                     "entity": f"Hub{hid}->Center{cid}",
    #                     "start": 1,
    #                     "end": 2,
    #                     "stage": "Hub->Center"
    #                 })
    # # Example for GA
    # elif ga_sol and "detailed_routes" in ga_sol:
    #     for route in ga_sol["detailed_routes"]:
    #         fid = route["farm_id"]
    #         hid = route["hub_id"]
    #         cid = route["center_id"]
    #         timeline_data.append({
    #             "entity": f"Farm{fid}->Hub{hid}",
    #             "start": 0,
    #             "end": 1,
    #             "stage": "Farm->Hub"
    #         })
    #         timeline_data.append({
    #             "entity": f"Hub{hid}->Center{cid}",
    #             "start": 1,
    #             "end": 2,
    #             "stage": "Hub->Center"
    #         })

    # if timeline_data:
    #     df_tl = pd.DataFrame(timeline_data)
    #     chart = alt.Chart(df_tl).mark_bar().encode(
    #         x='start:Q',
    #         x2='end:Q',
    #         y=alt.Y('entity:N', sort=alt.EncodingSortField(field='entity', order='ascending')),
    #         color='stage:N'
    #     ).properties(width=700, height=300)
    #     st.altair_chart(chart, use_container_width=True)
    # else:
    #     st.info("No timeline data to show. Generate or load solutions first.")

    # ----------------------------
    # 8) Compare Solutions
    # ----------------------------
    st.subheader("7) Compare Solutions")
    st.write(
        "Click the **Refresh Analytics** button below to compare the solutions "
        "stored in memory (Greedy, LP, GA) in terms of total cost, spoilage cost, etc."
    )
    if st.button("Refresh Analytics"):
        show_comparisons()


if __name__ == "__main__":
    main()
