#!/usr/bin/env python
"""
main.py
-------
Streamlit Dashboard for Agro-Route.

Usage:
    streamlit run main.py
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import json
import math

from data_input import (
    simulate_farms, simulate_storage_hubs,
    simulate_distribution_centers, simulate_vehicles
)
from distance_azure import build_distance_matrix
from greedy_kmeans import run_greedy_kmeans
from linear_programming import run_linear_program
from ga_mutation import run_genetic_algorithm

def display_node_map(farms, hubs, centers):
    """
    Displays a map with Farms, Hubs, Centers in different colors.
    """
    if not farms and not hubs and not centers:
        st.info("No data to display on map.")
        return

    layers = []
    # Farms => red
    if farms:
        df_f = pd.DataFrame({
            "id": [f["id"] for f in farms],
            "lat": [f["location"][0] for f in farms],
            "lon": [f["location"][1] for f in farms],
            "type": ["Farm"]*len(farms)
        })
        layer_farms = pdk.Layer(
            "ScatterplotLayer",
            data=df_f,
            get_position=["lon","lat"],
            get_radius=2500,
            get_fill_color=[255,0,0],
            pickable=True
        )
        layers.append(layer_farms)

    # Hubs => blue
    if hubs:
        df_h = pd.DataFrame({
            "id": [h["id"] for h in hubs],
            "lat": [h["location"][0] for h in hubs],
            "lon": [h["location"][1] for h in hubs],
            "type": ["Hub"]*len(hubs)
        })
        layer_hubs = pdk.Layer(
            "ScatterplotLayer",
            data=df_h,
            get_position=["lon","lat"],
            get_radius=3000,
            get_fill_color=[0,0,255],
            pickable=True
        )
        layers.append(layer_hubs)

    # Centers => green
    if centers:
        df_c = pd.DataFrame({
            "id": [c["id"] for c in centers],
            "lat": [c["location"][0] for c in centers],
            "lon": [c["location"][1] for c in centers],
            "type": ["Center"]*len(centers)
        })
        layer_centers = pdk.Layer(
            "ScatterplotLayer",
            data=df_c,
            get_position=["lon","lat"],
            get_radius=3500,
            get_fill_color=[0,255,0],
            pickable=True
        )
        layers.append(layer_centers)

    # center map
    lat0, lon0 = 28.7041, 77.1025
    if farms:
        lat0, lon0 = farms[0]["location"]
    elif hubs:
        lat0, lon0 = hubs[0]["location"]
    elif centers:
        lat0, lon0 = centers[0]["location"]

    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6, pitch=30)
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9")
    st.pydeck_chart(deck)

def display_distance_map(dist_dict, farms, hubs, centers):
    """
    Show lines for each pair for route_id=1, labeled with distance.
    """
    lines = []
    texts = []

    def get_location(node_type, node_id):
        if node_type == "farm":
            obj = next((f for f in farms if f["id"] == node_id), None)
            return obj["location"] if obj else None
        elif node_type == "hub":
            obj = next((h for h in hubs if h["id"] == node_id), None)
            return obj["location"] if obj else None
        elif node_type == "center":
            obj = next((c for c in centers if c["id"] == node_id), None)
            return obj["location"] if obj else None
        return None

    for (t1, id1, t2, id2), roads in dist_dict.items():
        if not roads:
            continue
        dist_km = None
        for rd in roads:
            if rd["route_id"] == 1:
                dist_km = rd["distance_km"]
                break
        if dist_km is None:
            continue

        loc1 = get_location(t1, id1)
        loc2 = get_location(t2, id2)
        if loc1 and loc2:
            lines.append({
                "path": [[loc1[1], loc1[0]], [loc2[1], loc2[0]]],
                "color": [128,128,128],
                "distance": dist_km
            })
            mid_lon = (loc1[1] + loc2[1])/2
            mid_lat = (loc1[0] + loc2[0])/2
            texts.append({
                "coordinates": [mid_lon, mid_lat],
                "text": f"{dist_km:.1f} km"
            })

    if not lines:
        st.info("No roads found to display.")
        return

    line_layer = pdk.Layer(
        "PathLayer",
        data=lines,
        get_path="path",
        get_color="color",
        width_min_pixels=2,
        pickable=True
    )
    text_layer = pdk.Layer(
        "TextLayer",
        data=texts,
        pickable=False,
        get_position="coordinates",
        get_text="text",
        get_size=16,
        get_color=[0,0,0],
        sizeUnits='meters',
        sizeScale=20,
        sizeMinPixels=12
    )

    # center
    lat0, lon0 = 28.7041, 77.1025
    if farms:
        lat0, lon0 = farms[0]["location"]
    elif hubs:
        lat0, lon0 = hubs[0]["location"]
    elif centers:
        lat0, lon0 = centers[0]["location"]

    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6, pitch=30)
    deck = pdk.Deck(layers=[line_layer, text_layer], initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/light-v9")
    st.pydeck_chart(deck)

def main():
    st.title("Agro-Route: Optimized Agricultural Supply Chain System")

    st.sidebar.title("1) Data Input Options")

    input_mode = st.sidebar.radio("Choose Input Mode:", ["Manual", "Simulate"])

    if "farms" not in st.session_state:
        st.session_state["farms"] = []
    if "hubs" not in st.session_state:
        st.session_state["hubs"] = []
    if "centers" not in st.session_state:
        st.session_state["centers"] = []
    if "vehicles" not in st.session_state:
        st.session_state["vehicles"] = []
    if "dist_dict" not in st.session_state:
        st.session_state["dist_dict"] = {}

    # Manual Input
    if input_mode == "Manual":
        st.sidebar.subheader("Enter Numbers of Each Entity:")
        num_farms = st.sidebar.number_input("Num Farms:", min_value=1, max_value=50, value=3)
        num_hubs = st.sidebar.number_input("Num Hubs:", min_value=1, max_value=20, value=2)
        num_centers = st.sidebar.number_input("Num Centers:", min_value=1, max_value=20, value=2)
        num_vehicles = st.sidebar.number_input("Num Vehicles:", min_value=1, max_value=10, value=3)

        if st.sidebar.button("Create Manual Forms"):
            # Create placeholders
            st.session_state["farms"] = []
            for i in range(num_farms):
                st.session_state["farms"].append({
                    "id": i,
                    "location": [28.7 + i*0.01, 77.1 + i*0.01],
                    "produce_quantity": 500,
                    "perishability_window": 3
                })

            st.session_state["hubs"] = []
            for i in range(num_hubs):
                st.session_state["hubs"].append({
                    "id": i,
                    "location": [28.6 + i*0.01, 77.0 + i*0.01],
                    "capacity": 2000,
                    "storage_cost_per_unit": 1.0,
                    "fixed_usage_cost": 1000
                })

            st.session_state["centers"] = []
            for i in range(num_centers):
                st.session_state["centers"].append({
                    "id": i,
                    "location": [28.9 + i*0.01, 77.2 + i*0.01],
                    "demand": 800,
                    "deadline": 4
                })

            st.session_state["vehicles"] = []
            for i in range(num_vehicles):
                typ = "small" if i < 2 else "large"
                st.session_state["vehicles"].append({
                    "id": i,
                    "type": typ,
                    "capacity": 1000 if typ=="small" else 3000,
                    "fixed_cost": 200 if typ=="small" else 350,
                    "variable_cost_per_distance": 0.8 if typ=="small" else 1.0
                })

    # Simulate
    if input_mode == "Simulate":
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

    st.subheader("3) Map of Farms, Hubs, Centers")
    display_node_map(st.session_state["farms"], st.session_state["hubs"], st.session_state["centers"])

    st.subheader("4) Build Distance Matrix")
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

    if st.session_state["dist_dict"]:
        st.write("Distance Matrix (sample):")
        sample_items = list(st.session_state["dist_dict"].items())[:10]
        for k,v in sample_items:
            st.write(f"{k}: {v}")

        st.subheader("Optional: Display Distance Map")
        if st.button("Show Distances on Map"):
            display_distance_map(st.session_state["dist_dict"],
                                 st.session_state["farms"],
                                 st.session_state["hubs"],
                                 st.session_state["centers"])

    st.subheader("5) Run Benchmarks / Optimization")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Greedy K-Means"):
            sol = run_greedy_kmeans(st.session_state["farms"],
                                    st.session_state["hubs"],
                                    st.session_state["centers"],
                                    st.session_state["vehicles"],
                                    st.session_state["dist_dict"],
                                    output_file="greedy_solution.json")
            st.session_state["greedy_solution"] = sol
            st.success("Greedy K-Means Completed!")

    with colB:
        if st.button("Linear Programming"):
            sol = run_linear_program(st.session_state["farms"],
                                     st.session_state["hubs"],
                                     st.session_state["centers"],
                                     st.session_state["vehicles"],
                                     st.session_state["dist_dict"],
                                     output_file="lp_solution.json")
            st.session_state["lp_solution"] = sol
            st.success("LP Completed!")

    with colC:
        if st.button("Genetic Algorithm"):
            sol = run_genetic_algorithm(st.session_state["farms"],
                                        st.session_state["hubs"],
                                        st.session_state["centers"],
                                        st.session_state["vehicles"],
                                        st.session_state["dist_dict"],
                                        pop_size=50,
                                        max_generations=50,
                                        output_file="ga_solution.json")
            st.session_state["ga_solution"] = sol
            st.success("Genetic Algorithm Completed!")

    st.subheader("6) Results & Visualization")

    def show_greedy_result(gsol):
        if not gsol:
            st.info("No Greedy solution found.")
            return
        st.write("**Greedy Solution JSON**:")
        st.json(gsol)

        # Simple route map for stage1 farm->hub lines
        lines = []
        for cl in gsol["clusters"]:
            hub_id = cl.get("hub_id")
            if hub_id is None:
                continue
            hub_obj = next((h for h in st.session_state["hubs"] if h["id"] == hub_id), None)
            if not hub_obj:
                continue
            hub_loc = hub_obj["location"]
            for cfarm in cl["cluster_farms"]:
                fid = cfarm["farm_id"]
                farm_obj = next((f for f in st.session_state["farms"] if f["id"] == fid), None)
                if farm_obj:
                    lines.append({
                        "path": [
                            [farm_obj["location"][1], farm_obj["location"][0]],
                            [hub_loc[1], hub_loc[0]]
                        ],
                        "color": [255, 0, 0]
                    })
        if lines:
            layer = pdk.Layer(
                "PathLayer",
                data=lines,
                get_path="path",
                get_color="color",
                width_min_pixels=3
            )
            lat0, lon0 = 28.7, 77.1
            if st.session_state["farms"]:
                lat0, lon0 = st.session_state["farms"][0]["location"]
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6, pitch=30),
                map_style="mapbox://styles/mapbox/light-v9"
            )
            st.pydeck_chart(deck)

    def show_lp_result(lpsol):
        if not lpsol:
            st.info("No LP solution found.")
            return
        st.write("**LP Solution JSON**:")
        st.json(lpsol)

        # route map
        lines = []
        for route in lpsol.get("detailed_routes", []):
            fid = route["farm_id"]
            hid = route["hub_id"]
            cid = route["center_id"]
            farm_obj = next((f for f in st.session_state["farms"] if f["id"] == fid), None)
            hub_obj = next((h for h in st.session_state["hubs"] if h["id"] == hid), None)
            cent_obj = next((c for c in st.session_state["centers"] if c["id"] == cid), None)
            if farm_obj and hub_obj:
                lines.append({
                    "path": [
                        [farm_obj["location"][1], farm_obj["location"][0]],
                        [hub_obj["location"][1], hub_obj["location"][0]]
                    ],
                    "color": [0, 0, 255]
                })
            if hub_obj and cent_obj:
                lines.append({
                    "path": [
                        [hub_obj["location"][1], hub_obj["location"][0]],
                        [cent_obj["location"][1], cent_obj["location"][0]]
                    ],
                    "color": [0, 255, 0]
                })
        if lines:
            layer = pdk.Layer("PathLayer", data=lines, get_path="path", get_color="color", width_min_pixels=3)
            lat0, lon0 = (28.7, 77.1)
            if st.session_state["farms"]:
                lat0, lon0 = st.session_state["farms"][0]["location"]
            deck = pdk.Deck(layers=[layer],
                            initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6, pitch=30),
                            map_style="mapbox://styles/mapbox/light-v9")
            st.pydeck_chart(deck)

        # Show table
        droutes = lpsol.get("detailed_routes", [])
        if droutes:
            st.write("**LP Route Assignments**:")
            df_dr = pd.DataFrame(droutes)
            st.dataframe(df_dr)

    def show_ga_result(ga_sol):
        if not ga_sol:
            st.info("No GA solution found.")
            return
        st.write("**GA Solution JSON**:")
        st.json(ga_sol)

        # We'll show each 1-unit route
        lines = []
        for route in ga_sol.get("detailed_routes", []):
            fid = route["farm_id"]
            hid = route["hub_id"]
            cid = route["center_id"]
            farm_obj = next((f for f in st.session_state["farms"] if f["id"] == fid), None)
            hub_obj = next((h for h in st.session_state["hubs"] if h["id"] == hid), None)
            cent_obj = next((c for c in st.session_state["centers"] if c["id"] == cid), None)

            if farm_obj and hub_obj and route["dist_farm_hub"] is not None:
                lines.append({
                    "path": [
                        [farm_obj["location"][1], farm_obj["location"][0]],
                        [hub_obj["location"][1], hub_obj["location"][0]]
                    ],
                    "color": [255, 0, 255]
                })
            if hub_obj and cent_obj and route["dist_hub_center"] is not None:
                lines.append({
                    "path": [
                        [hub_obj["location"][1], hub_obj["location"][0]],
                        [cent_obj["location"][1], cent_obj["location"][0]]
                    ],
                    "color": [255, 165, 0] # orange
                })
        if lines:
            layer = pdk.Layer("PathLayer", data=lines, get_path="path", get_color="color", width_min_pixels=3)
            lat0, lon0 = (28.7, 77.1)
            if st.session_state["farms"]:
                lat0, lon0 = st.session_state["farms"][0]["location"]
            deck = pdk.Deck(layers=[layer],
                            initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6, pitch=30),
                            map_style="mapbox://styles/mapbox/light-v9")
            st.pydeck_chart(deck)

        dr = ga_sol.get("detailed_routes", [])
        if dr:
            st.write("**GA Detailed Routes**:")
            df_dr = pd.DataFrame(dr)
            st.dataframe(df_dr)

    tab1, tab2, tab3 = st.tabs(["Greedy", "LP", "GA"])
    with tab1:
        show_greedy_result(st.session_state.get("greedy_solution"))
    with tab2:
        show_lp_result(st.session_state.get("lp_solution"))
    with tab3:
        show_ga_result(st.session_state.get("ga_solution"))

    # Optional timeline
    st.subheader("7) Timeline (Concept)")

    timeline_data = []
    # If we have an LP solution with details:
    lp_sol = st.session_state.get("lp_solution")
    if lp_sol and "detailed_routes" in lp_sol:
        for route in lp_sol["detailed_routes"]:
            fid = route["farm_id"]
            hid = route["hub_id"]
            cid = route["center_id"]
            timeline_data.append({
                "entity": f"Farm{fid}->Hub{hid}",
                "start": 0,
                "end": 1,
                "stage": "Farm->Hub"
            })
            timeline_data.append({
                "entity": f"Hub{hid}->Center{cid}",
                "start": 1,
                "end": 2,
                "stage": "Hub->Center"
            })
    elif st.session_state.get("greedy_solution"):
        for cl in st.session_state["greedy_solution"].get("clusters", []):
            hid = cl.get("hub_id")
            if hid is not None:
                for cf in cl["cluster_farms"]:
                    timeline_data.append({
                        "entity": f"Farm{cf['farm_id']}->Hub{hid}",
                        "start": 0,
                        "end": 1,
                        "stage": "Farm->Hub"
                    })
                timeline_data.append({
                    "entity": f"Hub{hid}->Center(?)",
                    "start": 1,
                    "end": 2,
                    "stage": "Hub->Center"
                })
    elif st.session_state.get("ga_solution"):
        # We'll just do farm->hub->center in 2 steps for each 1-unit
        for route in st.session_state["ga_solution"].get("detailed_routes", []):
            fid = route["farm_id"]
            hid = route["hub_id"]
            cid = route["center_id"]
            timeline_data.append({
                "entity": f"Farm{fid}->Hub{hid}",
                "start": 0,
                "end": 1,
                "stage": "Farm->Hub"
            })
            timeline_data.append({
                "entity": f"Hub{hid}->Center{cid}",
                "start": 1,
                "end": 2,
                "stage": "Hub->Center"
            })

    if timeline_data:
        df_tl = pd.DataFrame(timeline_data)
        chart = alt.Chart(df_tl).mark_bar().encode(
            x='start:Q',
            x2='end:Q',
            y=alt.Y('entity:N', sort=alt.EncodingSortField(field='entity', order='ascending')),
            color='stage:N'
        ).properties(width=700, height=300)
        st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
