#!/usr/bin/env python
"""
main.py
-------
A Streamlit dashboard offering:
  - Data input (manual or simulate) for Farms, Hubs, Centers, Vehicles
  - Distance matrix generation with Azure Maps or fallback
  - Running two benchmarks: Greedy K-Means and LP (CPLEX)
  - Visualization:
    * Maps of routes (pydeck)
    * Timeline/Gantt chart for produce movement
    * Cost/Profit summary
  - Comparison of results

Usage:
    streamlit run main.py
"""

import streamlit as st
import pandas as pd
import json
import os

import altair as alt
import pydeck as pdk

from data_input import (
    simulate_farms, simulate_storage_hubs, simulate_distribution_centers, simulate_vehicles,
    manual_input_farms, manual_input_storage_hubs, manual_input_distribution_centers, manual_input_vehicles
)
from distance_azure import build_distance_matrix
from greedy_kmeans import run_greedy_kmeans
from linear_programming import run_linear_program

st.set_page_config(page_title="Agro-Route Dashboard", layout="wide")


def main():
    st.title("Agro-Route: Optimized Agricultural Supply Chain Management")

    st.sidebar.title("Input Options")
    # 1) Choose to simulate or manually input data
    data_mode = st.sidebar.radio("Data Mode:", ["Simulate", "Manual"])

    if st.sidebar.button("Load / Generate Data"):
        if data_mode == "Simulate":
            with st.spinner("Generating random data..."):
                farms = simulate_farms(num_farms=st.sidebar.slider("Number of Farms", 1, 50, 5))
                hubs = simulate_storage_hubs(num_hubs=st.sidebar.slider("Number of Hubs", 1, 20, 3))
                centers = simulate_distribution_centers(num_centers=st.sidebar.slider("Number of Centers", 1, 20, 2))
                vehicles = simulate_vehicles(num_small=st.sidebar.slider("Small Vehicles", 1, 5, 2),
                                             num_large=st.sidebar.slider("Large Vehicles", 0, 5, 1))
        else:
            with st.spinner("Loading manual data..."):
                farms = manual_input_farms()
                hubs = manual_input_storage_hubs()
                centers = manual_input_distribution_centers()
                vehicles = manual_input_vehicles()
        st.success("Data Loaded.")
        st.session_state["farms"] = farms
        st.session_state["hubs"] = hubs
        st.session_state["centers"] = centers
        st.session_state["vehicles"] = vehicles

    # Show current data in memory if it exists
    if "farms" in st.session_state:
        st.subheader("Current Data Summary")
        farms = st.session_state["farms"]
        hubs = st.session_state["hubs"]
        centers = st.session_state["centers"]
        vehicles = st.session_state["vehicles"]

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Farms**")
            st.dataframe(pd.DataFrame(farms))
            st.write("**Storage Hubs**")
            st.dataframe(pd.DataFrame(hubs))
        with col2:
            st.write("**Distribution Centers**")
            st.dataframe(pd.DataFrame(centers))
            st.write("**Vehicles**")
            st.dataframe(pd.DataFrame(vehicles))

        # 2) Build distance matrix
        st.subheader("Generate Distance Matrix")
        use_azure = st.checkbox("Use Azure Maps API? (Requires valid key in distance_azure.py)")
        if st.button("Build Distance Matrix"):
            with st.spinner("Building distance matrix..."):
                dist_dict = build_distance_matrix(farms, hubs, centers, use_azure=use_azure)
            st.session_state["dist_dict"] = dist_dict
            st.success("Distance matrix generated.")
            st.write("Example entries:")
            small_sample = list(dist_dict.items())[:10]
            st.write(small_sample)

        # 3) Benchmarks: Greedy + LP
        if "dist_dict" in st.session_state:
            st.subheader("Run Benchmarks / Optimization")
            if st.button("Run Greedy K-Means"):
                with st.spinner("Running Greedy K-Means..."):
                    greedy_sol = run_greedy_kmeans(farms, hubs, centers, vehicles, st.session_state["dist_dict"],
                                                   output_file="greedy_solution.json")
                st.session_state["greedy_solution"] = greedy_sol
                st.success("Greedy K-Means done. Results stored in greedy_solution.json")

            if st.button("Run Linear Programming (CPLEX)"):
                with st.spinner("Running LP..."):
                    lp_sol = run_linear_program(farms, hubs, centers, vehicles, st.session_state["dist_dict"],
                                                output_file="lp_solution.json")
                st.session_state["lp_solution"] = lp_sol
                st.success("LP done. Results stored in lp_solution.json")

            # 4) Visualization
            st.subheader("Visualization of Solutions")

            def visualize_solution(solution, name):
                """
                For either greedy or lp solution, we attempt to draw a map of routes
                and a timeline chart for the movement.
                """
                st.write(f"### {name} Solution")

                if solution is None:
                    st.info(f"No {name} solution found.")
                    return

                # Try to parse route info
                if solution.get("method", "").startswith("Greedy"):
                    # We have 'clusters' structure with cluster_farms, hub_id, etc.
                    lines = []
                    for cl in solution["clusters"]:
                        hub_id = cl["hub_id"]
                        if hub_id is None: 
                            continue
                        hub_loc = next((h["location"] for h in hubs if h["id"] == hub_id), None)
                        # For each farm in cluster
                        for cf in cl["cluster_farms"]:
                            fid = cf["farm_id"]
                            farm_loc = next((f["location"] for f in farms if f["id"] == fid), None)
                            if farm_loc and hub_loc:
                                lines.append({
                                    "path": [[farm_loc[1], farm_loc[0]], [hub_loc[1], hub_loc[0]]],
                                    "color": [255, 0, 0]
                                })
                        # Then hub->center
                        # We assume we pick nearest center in the code, but not stored as a direct reference.
                        # For simplicity, skip or try to guess the center with demand > 0. 
                        # The code in run_greedy_kmeans actually modifies the center's demand in place.
                        # This is a demonstration, so we won't reconstruct that route easily.
                    if lines:
                        layer = pdk.Layer(
                            "PathLayer",
                            data=lines,
                            get_path="path",
                            get_color="color",
                            width_min_pixels=4
                        )
                        # center map
                        init_lat = farms[0]["location"][0]
                        init_lon = farms[0]["location"][1]
                        view_state = pdk.ViewState(latitude=init_lat, longitude=init_lon, zoom=6, pitch=30)
                        deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                                        map_style="mapbox://styles/mapbox/light-v9")
                        st.pydeck_chart(deck)

                    st.json(solution)

                else:
                    # For LP: "stage1_assignments" and "stage2_assignments"
                    lines = []
                    # stage1
                    for a in solution.get("stage1_assignments", []):
                        fid = a["farm_id"]
                        hid = a["hub_id"]
                        floc = next((f["location"] for f in farms if f["id"] == fid), None)
                        hloc = next((h["location"] for h in hubs if h["id"] == hid), None)
                        if floc and hloc:
                            lines.append({
                                "path": [[floc[1], floc[0]], [hloc[1], hloc[0]]],
                                "color": [0, 0, 255]
                            })
                    # stage2
                    for a in solution.get("stage2_assignments", []):
                        hid = a["hub_id"]
                        cid = a["center_id"]
                        hloc = next((h["location"] for h in hubs if h["id"] == hid), None)
                        cloc = next((c["location"] for c in centers if c["id"] == cid), None)
                        if hloc and cloc:
                            lines.append({
                                "path": [[hloc[1], hloc[0]], [cloc[1], cloc[0]]],
                                "color": [0, 255, 0]
                            })

                    if lines:
                        layer = pdk.Layer(
                            "PathLayer",
                            data=lines,
                            get_path="path",
                            get_color="color",
                            width_min_pixels=4
                        )
                        init_lat = farms[0]["location"][0]
                        init_lon = farms[0]["location"][1]
                        view_state = pdk.ViewState(latitude=init_lat, longitude=init_lon, zoom=6, pitch=30)
                        deck = pdk.Deck(layers=[layer], initial_view_state=view_state,
                                        map_style="mapbox://styles/mapbox/light-v9")
                        st.pydeck_chart(deck)
                    st.json(solution)

                # Build a timeline / Gantt demonstration
                # We'll just create a dummy timeline: farm->hub from time=0 to 1, hub->center from time=1 to 2
                # Real times would be computed from distance / speed, plus waiting times, etc.
                # This is just to show how a Gantt can appear.
                if "clusters" in solution:  # Greedy structure
                    timeline_data = []
                    for cl in solution["clusters"]:
                        for cfarm in cl["cluster_farms"]:
                            f_id = cfarm["farm_id"]
                            start_t = 0
                            end_t = 1
                            timeline_data.append({
                                "entity": f"Farm{f_id}->Hub{cl['hub_id']}",
                                "start": start_t,
                                "end": end_t,
                                "stage": "Transport to Hub"
                            })
                        # Then hub->center is next stage
                        start_t = 1
                        end_t = 2
                        timeline_data.append({
                            "entity": f"Hub{cl['hub_id']}->Center(unknown)",
                            "start": start_t,
                            "end": end_t,
                            "stage": "Transport to Center"
                        })
                else:
                    # LP structure
                    timeline_data = []
                    for a in solution.get("stage1_assignments", []):
                        start_t = 0
                        end_t = 1
                        timeline_data.append({
                            "entity": f"Farm{a['farm_id']}->Hub{a['hub_id']}",
                            "start": start_t,
                            "end": end_t,
                            "stage": "Transport to Hub"
                        })
                    for a in solution.get("stage2_assignments", []):
                        start_t = 1
                        end_t = 2
                        timeline_data.append({
                            "entity": f"Hub{a['hub_id']}->Center{a['center_id']}",
                            "start": start_t,
                            "end": end_t,
                            "stage": "Transport to Center"
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


            colA, colB = st.columns(2)
            with colA:
                if "greedy_solution" in st.session_state:
                    visualize_solution(st.session_state["greedy_solution"], "Greedy K-Means")
                else:
                    st.info("Run the Greedy K-Means solution first to visualize.")

            with colB:
                if "lp_solution" in st.session_state:
                    visualize_solution(st.session_state["lp_solution"], "Linear Programming")
                else:
                    st.info("Run the LP solution first to visualize.")


if __name__ == "__main__":
    main()
