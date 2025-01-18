#!/usr/bin/env python
"""
run_comparison.py
-----------------
Demonstration script to:
 1) Generate or load data
 2) Build distance matrix
 3) Run Greedy K-Means solution
 4) Run LP solution
 5) Compare results => write to "comparison.json"
"""

import json
import pandas as pd
import os

from data_input import simulate_farms, simulate_storage_hubs, simulate_distribution_centers, simulate_vehicles
from distance_azure import build_distance_matrix
from greedy_kmeans import run_greedy_kmeans
from linear_programming import run_linear_program

def main():
    # Example pipeline
    # 1) Simulate data
    farms = simulate_farms(num_farms=5)
    hubs = simulate_storage_hubs(num_hubs=2)
    centers = simulate_distribution_centers(num_centers=2)
    vehicles = simulate_vehicles(num_small=2, num_large=1)

    # 2) Build distance dictionary
    dist_dict = build_distance_matrix(farms, hubs, centers, use_azure=False)

    # 3) Run greedy
    greedy_sol = run_greedy_kmeans(farms, hubs, centers, vehicles, dist_dict, output_file="greedy_solution.json")

    # 4) Run LP
    lp_sol = run_linear_program(farms, hubs, centers, vehicles, dist_dict, output_file="lp_solution.json")

    # 5) Compare
    data = []
    if greedy_sol:
        data.append({
            "method": greedy_sol["method"],
            "total_cost": greedy_sol.get("total_cost", 0),
            "total_spoilage": greedy_sol.get("total_spoilage", 0),
            "delivered_on_time": greedy_sol.get("total_delivered_on_time", 0),
        })
    if lp_sol:
        data.append({
            "method": lp_sol["method"],
            "total_cost": lp_sol.get("objective_value", 0),
            "total_spoilage": lp_sol.get("total_spoilage", 0),
            "delivered_on_time": lp_sol.get("delivered_on_time", 0)
        })

    df = pd.DataFrame(data)
    df.to_json("comparison.json", orient="records", indent=4)
    print(df)

if __name__ == "__main__":
    main()
