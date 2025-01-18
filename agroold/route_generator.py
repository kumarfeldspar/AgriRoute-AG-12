#!/usr/bin/env python
"""
route_generator.py
------------------
Generates a two-stage route solution:
  Stage 1: Farms -> Hubs
  Stage 2: Hubs -> Distribution Centers

Uses similar K-means approach (with cluster radii based on "goods type")
for demonstration.

Reads:
  - farms.json
  - hubs.json
  - centers.json

Outputs:
  - fulldf.json (detailed route info for both stages)
  - destdf_stage1.json, destdf_stage2.json (optional for debugging)
  - stage1_src.json, stage2_src.json (optional)
"""

import numpy as np
import pandas as pd
import requests
import random
import time
import json
from sklearn.cluster import KMeans

# ------------------ PARAMETERS ------------------
np.random.seed(1)
thresh = 0.4
vthresh = 1

# Emission factor and average speed (same as your single-stage code)
EMISSION_FACTOR = 0.21
AVG_SPEED = 40

AZURE_API_KEY = "4j0DlBhKmXzbG2frDTurBFfAq6NPysfILisACKoao83EcEjvmEobJQQJ99BAACYeBjFeh7q1AAAgAZMP3Pj2"  # Replace!

# "Goods types" concept used just for demonstration
goods_types = ["Perishable", "Frozen", "Non-Perishable"]
type_radius_map = {
    "Perishable": 100000,
    "Frozen": 200000,
    "Non-Perishable": 300000
}
type_priority_map = {
    "Perishable": 3,
    "Frozen": 2,
    "Non-Perishable": 1
}

# ---------- HELPER FUNCTIONS ----------
def get_traffic_and_weather():
    time.sleep(0.02)
    return {
        "trafficLevel": random.choice(["low", "moderate", "high"]),
        "weather": random.choice(["clear", "rain", "fog", "cloudy"])
    }

def compute_carbon_emission(distance_m):
    distance_km = distance_m / 1000.0
    return round(distance_km * EMISSION_FACTOR, 3)

def compute_eta(distance_m):
    distance_km = distance_m / 1000.0
    eta_minutes = (distance_km / AVG_SPEED) * 60
    return round(eta_minutes, 1)

def get_route_query_points(points, src, is_summary=True):
    """
    Build query for Azure Maps route from src -> points -> src to close loop.
    """
    s = f"{src[0]},{src[1]}:"
    for i, p in enumerate(points):
        if i < 145:
            s += f"{p[0]},{p[1]}:"
    # close loop
    s += f"{src[0]},{src[1]}"
    route_repr = "summaryOnly" if is_summary else "polyline"
    return s, route_repr

def call_route_api(query, route_repr="summaryOnly"):
    full_url = (
        f"https://atlas.microsoft.com/route/directions/json"
        f"?subscription-key={AZURE_API_KEY}"
        f"&api-version=1.0"
        f"&query={query}"
        f"&routeType=shortest"
        f"&computeBestOrder=true"
        f"&travelMode=car"
        f"&routeRepresentation={route_repr}"
    )
    resp = requests.get(full_url).json()
    return resp

def build_dataframe_for_route(route_info, stage):
    """
    Convert route dictionary into a DataFrame row.
    route_info = {
      'src_idx': ...
      'veh_idx': ...
      'path': [...],
      ...
    }
    """
    path_points = pd.DataFrame(route_info["path"])
    df = pd.DataFrame({
        "stage": [stage],
        "src_idx": [route_info["src_idx"]],
        "veh_idx": [route_info["veh_idx"]],
        "path": [path_points.reset_index()[["longitude","latitude"]].values.tolist()],
        "length_m": [route_info["length"]],
        "emissions_kgCO2": [route_info["emissions"]],
        "eta_minutes": [route_info["eta_minutes"]],
        "traffic": [route_info["traffic"]],
        "weather": [route_info["weather"]],
        "goodsType": [route_info["goodsType"]],
        "priority": [route_info["priority"]]
    })
    return df


# ---------- MAIN 2-STAGE LOGIC ----------
def main():
    # 1) Read dataset
    with open("farms.json") as f:
        farms = json.load(f)
    with open("hubs.json") as f:
        hubs = json.load(f)
    with open("centers.json") as f:
        centers = json.load(f)

    # Convert to NumPy for clustering
    farm_coords = np.array([f["location"] for f in farms])
    hub_coords = np.array([h["location"] for h in hubs])
    center_coords = np.array([c["location"] for c in centers])

    # For demonstration, assign a random goods type to each "farm"
    # (In a real scenario, you'd interpret each farm's produce to decide type)
    farm_goods_types = [random.choice(goods_types) for _ in range(len(farms))]

    # ---------- STAGE 1: Farms -> Hubs ----------
    # We'll do a K-means to cluster each farm to a single hub
    # Then do near/far approach for each hub to get sub-vehicles
    # (This is simplistic; real logic might require capacity checks, etc.)

    # 1A) K-means global to find which hub each farm belongs to
    #     We'll make #clusters = # hubs
    combined_coords = np.vstack([farm_coords, hub_coords])
    kmeans_farmhub = KMeans(n_clusters=len(hubs), random_state=0).fit(combined_coords)
    # The first portion belongs to farms
    farm_labels = kmeans_farmhub.labels_[: len(farm_coords)]
    # Each farm i is assigned to the hub with label farm_labels[i]

    # For each hub label, gather farm coordinates
    stage1_clusters = []
    for hub_label in range(len(hubs)):
        cluster_points = farm_coords[farm_labels == hub_label]
        stage1_clusters.append(cluster_points)

    # We'll pretend each "hub_label" is "src_idx" 
    # Then generate near/far inside each cluster for multiple vehicles
    def near_check(p, ref):
        return np.linalg.norm(p - ref, axis=1) < 0.4

    # Build final DataFrame for stage1
    stage1_df = pd.DataFrame()

    for h_idx, cluster_points in enumerate(stage1_clusters):
        # Goods type from the farm's "dominant" or random pick
        # We'll pick the first farm's type, or a random from that cluster
        if len(cluster_points) == 0:
            continue
        random_good = random.choice(farm_goods_types)  # simplistic

        # near/far approach
        hub_loc = hub_coords[h_idx]  # the location of the hub
        dist = np.linalg.norm(cluster_points - hub_loc, axis=1)
        mask = dist < 0.4
        near_points = cluster_points[mask]
        far_points = cluster_points[~mask]

        sub_clusters = [near_points, far_points]
        # For each sub_cluster => route
        for v_idx, sub_pts in enumerate(sub_clusters):
            if len(sub_pts) == 0:
                continue
            # build route
            query, route_repr = get_route_query_points(sub_pts, hub_loc, is_summary=False)
            resp = call_route_api(query, route_repr)
            route_summary = resp["routes"][0]["summary"]
            distance_m = route_summary["lengthInMeters"]

            route_data = {}
            route_data["stage"] = 1
            route_data["src_idx"] = h_idx  # which hub
            route_data["veh_idx"] = v_idx
            route_data["path"] = []
            route_data["length"] = distance_m
            route_data["emissions"] = compute_carbon_emission(distance_m)
            route_data["eta_minutes"] = compute_eta(distance_m)

            tw = get_traffic_and_weather()
            route_data["traffic"] = tw["trafficLevel"]
            route_data["weather"] = tw["weather"]

            # path
            route_points = []
            for leg in resp["routes"][0]["legs"]:
                route_points += leg["points"]
            route_data["path"] = route_points

            # pick goods type
            route_data["goodsType"] = random_good
            route_data["priority"] = type_priority_map[random_good]

            df_temp = build_dataframe_for_route(route_data, stage=1)
            stage1_df = pd.concat([stage1_df, df_temp], ignore_index=True)

    # ---------- STAGE 2: Hubs -> Centers ----------
    # Similar approach but cluster each hub to a center
    # We do a K-means for hub_coords + center_coords => #clusters = len(centers)
    combined_hub_centers = np.vstack([hub_coords, center_coords])
    kmeans_hubcenter = KMeans(n_clusters=len(centers), random_state=0).fit(combined_hub_centers)
    hub_labels = kmeans_hubcenter.labels_[: len(hub_coords)]  # for each hub

    stage2_clusters = []
    for c_label in range(len(centers)):
        cluster_points = hub_coords[hub_labels == c_label]
        stage2_clusters.append(cluster_points)

    stage2_df = pd.DataFrame()
    for c_idx, cluster_points in enumerate(stage2_clusters):
        # We'll pick a random goods type for each cluster
        # Real logic might combine farm's goods -> hub -> center
        random_good = random.choice(goods_types)

        # near/far approach for "hub -> center"
        center_loc = center_coords[c_idx]
        dist = np.linalg.norm(cluster_points - center_loc, axis=1)
        mask = dist < 0.4
        near_points = cluster_points[mask]
        far_points = cluster_points[~mask]

        sub_clusters = [near_points, far_points]
        for v_idx, sub_pts in enumerate(sub_clusters):
            if len(sub_pts) == 0:
                continue
            query, route_repr = get_route_query_points(sub_pts, center_loc, is_summary=False)
            resp = call_route_api(query, route_repr)
            route_summary = resp["routes"][0]["summary"]
            distance_m = route_summary["lengthInMeters"]

            route_data = {}
            route_data["stage"] = 2
            route_data["src_idx"] = c_idx  # which center
            route_data["veh_idx"] = v_idx
            route_data["length"] = distance_m
            route_data["emissions"] = compute_carbon_emission(distance_m)
            route_data["eta_minutes"] = compute_eta(distance_m)

            tw = get_traffic_and_weather()
            route_data["traffic"] = tw["trafficLevel"]
            route_data["weather"] = tw["weather"]

            route_points = []
            for leg in resp["routes"][0]["legs"]:
                route_points += leg["points"]
            route_data["path"] = route_points

            route_data["goodsType"] = random_good
            route_data["priority"] = type_priority_map[random_good]

            df_temp = build_dataframe_for_route(route_data, stage=2)
            stage2_df = pd.concat([stage2_df, df_temp], ignore_index=True)

    # Merge both stages
    fulldf = pd.concat([stage1_df, stage2_df], ignore_index=True)
    fulldf.to_json("fulldf.json", orient="records")
    print("Two-stage route generation complete. Saved to fulldf.json")

if __name__ == "__main__":
    main()
