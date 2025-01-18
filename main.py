#!/usr/bin/env python
"""
main.py
-------
Streamlit UI for "AgriRoute – Optimized Multi-Stage Agricultural Supply Chain Design".

Usage:
    streamlit run main.py
"""

import streamlit as st
import json
import pandas as pd
import pydeck as pdk
import os

st.set_page_config(page_title="AgriRoute Multi-Stage Demo", layout="wide")

@st.cache_data
def load_data():
    farms, hubs, centers, vehicles = [], [], [], []
    if os.path.exists("farms.json"):
        farms = json.load(open("farms.json"))
    if os.path.exists("hubs.json"):
        hubs = json.load(open("hubs.json"))
    if os.path.exists("centers.json"):
        centers = json.load(open("centers.json"))
    if os.path.exists("vehicles.json"):
        vehicles = json.load(open("vehicles.json"))
    return farms, hubs, centers, vehicles

@st.cache_data
def load_solutions():
    sol_baseline, sol_novel = None, None
    if os.path.exists("baseline_solution.json"):
        sol_baseline = json.load(open("baseline_solution.json"))
    if os.path.exists("novel_solution.json"):
        sol_novel = json.load(open("novel_solution.json"))
    return sol_baseline, sol_novel

@st.cache_data
def load_comparison():
    if os.path.exists("comparison.json"):
        return pd.read_json("comparison.json", orient="records")
    return pd.DataFrame()

def overview_map(farms, hubs, centers):
    st.header("1) Overview of Farms, Hubs, Centers")
    layers = []
    # Farms
    if farms:
        df_farms = pd.DataFrame({
            "id": [f["id"] for f in farms],
            "lat": [f["location"][0] for f in farms],
            "lon": [f["location"][1] for f in farms],
        })
        farm_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_farms,
            get_position=["lon", "lat"],
            get_radius=4000,
            get_fill_color=[255, 255, 0],
            pickable=True
        )
        layers.append(farm_layer)

    # Hubs
    if hubs:
        df_hubs = pd.DataFrame({
            "id": [h["id"] for h in hubs],
            "lat": [h["location"][0] for h in hubs],
            "lon": [h["location"][1] for h in hubs],
        })
        hub_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_hubs,
            get_position=["lon", "lat"],
            get_radius=5000,
            get_fill_color=[0, 255, 255],
            pickable=True
        )
        layers.append(hub_layer)

    # Centers
    if centers:
        df_centers = pd.DataFrame({
            "id": [c["id"] for c in centers],
            "lat": [c["location"][0] for c in centers],
            "lon": [c["location"][1] for c in centers],
        })
        center_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_centers,
            get_position=["lon", "lat"],
            get_radius=5500,
            get_fill_color=[255, 0, 255],
            pickable=True
        )
        layers.append(center_layer)

    if len(farms) > 0:
        init_lat = farms[0]["location"][0]
        init_lon = farms[0]["location"][1]
    else:
        init_lat, init_lon = 28.7041, 77.1025

    view_state = pdk.ViewState(latitude=init_lat, longitude=init_lon, zoom=6, pitch=30)
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        layers=layers,
        initial_view_state=view_state
    )
    st.pydeck_chart(deck)
    st.caption("Farms=yellow, Hubs=cyan, Centers=magenta")

def visualize_baseline_solution(baseline_solution, farms, hubs, centers):
    st.subheader("Baseline Solution Visualization")

    # We'll just show stage1 (farm->hub) and stage2 (hub->center) as lines
    lines = []
    # farm->hub
    for r in baseline_solution["stage1_routes"]:
        if r["hub_id"] is None:
            continue
        fx, fy = next((f["location"] for f in farms if f["id"]==r["farm_id"]), (None,None))
        hx, hy = next((h["location"] for h in hubs if h["id"]==r["hub_id"]), (None,None))
        if fx is not None:
            lines.append({
                "path": [[fy, fx], [hy, hx]],  # note [lon, lat]
                "color": [255, 0, 0]
            })

    # hub->center
    for r in baseline_solution["stage2_routes"]:
        if r["center_id"] is None:
            continue
        hx, hy = next((h["location"] for h in hubs if h["id"]==r["hub_id"]), (None,None))
        cx, cy = next((c["location"] for c in centers if c["id"]==r["center_id"]), (None,None))
        if hx is not None:
            lines.append({
                "path": [[hy, hx], [cy, cx]],
                "color": [0, 255, 0]
            })

    if len(lines)>0:
        path_layer = pdk.Layer(
            "PathLayer",
            data=lines,
            get_path="path",
            get_color="color",
            width_min_pixels=4
        )
        init_lat, init_lon = (lines[0]["path"][0][1], lines[0]["path"][0][0])
        view_state = pdk.ViewState(latitude=init_lat, longitude=init_lon, zoom=6, pitch=30)
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            layers=[path_layer],
            initial_view_state=view_state
        )
        st.pydeck_chart(deck)
    else:
        st.info("No routes available to visualize in baseline solution.")

    st.json(baseline_solution)

def visualize_novel_solution(novel_solution, farms, hubs, centers):
    st.subheader("Novel Solution Visualization")

    lines = []
    # stage1
    for r in novel_solution["stage1_routes"]:
        fid = r["farm_id"]
        hid = r["hub_id"]
        fx, fy = next((f["location"] for f in farms if f["id"]==fid), (None,None))
        hx, hy = next((h["location"] for h in hubs if h["id"]==hid), (None,None))
        if fx is not None:
            lines.append({
                "path": [[fy, fx], [hy, hx]],
                "color": [255, 100, 100]
            })
    # stage2
    for r in novel_solution["stage2_routes"]:
        hid = r["hub_id"]
        cid = r["center_id"]
        hx, hy = next((h["location"] for h in hubs if h["id"]==hid), (None,None))
        cx, cy = next((c["location"] for c in centers if c["id"]==cid), (None,None))
        if hx is not None:
            lines.append({
                "path": [[hy, hx], [cy, cx]],
                "color": [100, 255, 100]
            })

    if len(lines)>0:
        path_layer = pdk.Layer(
            "PathLayer",
            data=lines,
            get_path="path",
            get_color="color",
            width_min_pixels=4
        )
        init_lat, init_lon = (lines[0]["path"][0][1], lines[0]["path"][0][0])
        view_state = pdk.ViewState(latitude=init_lat, longitude=init_lon, zoom=6, pitch=30)
        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            layers=[path_layer],
            initial_view_state=view_state
        )
        st.pydeck_chart(deck)
    else:
        st.info("No routes available in novel solution.")

    st.json(novel_solution)

def performance_metrics():
    st.header("Performance Comparison")
    comp_df = load_comparison()
    if comp_df.empty:
        st.info("No comparison.json found. Please run performance_analysis.py first.")
        return
    st.dataframe(comp_df)

def main():
    st.title("AgriRoute – Multi-Stage Agricultural Supply Chain Demo")

    farms, hubs, centers, vehicles = load_data()
    sol_baseline, sol_novel = load_solutions()

    # 1) Overview
    overview_map(farms, hubs, centers)

    # 2) Pick a solution to visualize
    st.header("2) Solution Visualization")
    solution_choice = st.selectbox("Select a solution to view:", ["None", "Baseline", "Novel"])
    if solution_choice == "Baseline":
        if sol_baseline:
            visualize_baseline_solution(sol_baseline, farms, hubs, centers)
        else:
            st.error("baseline_solution.json not found.")
    elif solution_choice == "Novel":
        if sol_novel:
            visualize_novel_solution(sol_novel, farms, hubs, centers)
        else:
            st.error("novel_solution.json not found.")

    # 3) Performance
    performance_metrics()

    st.header("4) Potential Disruption & Re-Optimization Demo (Concept)")
    st.write("In a real system, you'd detect a new road closure or changed cost, update `disruptions.json`, and re-run the solver. The new solution can then be visualized here. This demonstration only outlines the approach.")

    st.write("Use the left-hand menu to navigate additional expansions or modules.")

if __name__ == "__main__":
    main()
