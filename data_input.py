#!/usr/bin/env python
"""
data_input.py
-------------
Provides functions to SIMULATE data for:
 - Farms
 - Storage Hubs
 - Distribution Centers
 - Vehicles

Manual input is handled interactively in `main.py` with Streamlit forms.
"""

import random
import numpy as np

def simulate_farms(num_farms=5, seed=42):
    """
    Simulate random farm data:
      - id
      - location (lat, lon)
      - produce_quantity
      - perishability_window
    """
    random.seed(seed)
    np.random.seed(seed)

    # Center around some region (e.g., near Delhi)
    center_lat, center_lon = 28.7041, 77.1025
    results = []
    for i in range(num_farms):
        # Random shift in location
        lat_shift = np.random.normal(0, 0.15)
        lon_shift = np.random.normal(0, 0.15)
        produce_qty = random.randint(200, 1000)
        perish_window = random.randint(1, 5)

        results.append({
            "id": i,
            "location": [center_lat + lat_shift, center_lon + lon_shift],
            "produce_quantity": produce_qty,
            "perishability_window": perish_window
        })

    return results

def simulate_storage_hubs(num_hubs=2, seed=100):
    """
    Simulate random storage hubs:
      - id
      - location (lat, lon)
      - capacity
      - storage_cost_per_unit
      - fixed_usage_cost
    """
    random.seed(seed)
    np.random.seed(seed)

    center_lat, center_lon = 28.6, 77.0
    results = []
    for i in range(num_hubs):
        lat_shift = np.random.normal(0, 0.1)
        lon_shift = np.random.normal(0, 0.1)
        capacity = random.randint(1500, 8000)
        storage_cost = round(random.uniform(0.5, 2.0), 2)
        fixed_cost = random.randint(500, 2000)

        results.append({
            "id": i,
            "location": [center_lat + lat_shift, center_lon + lon_shift],
            "capacity": capacity,
            "storage_cost_per_unit": storage_cost,
            "fixed_usage_cost": fixed_cost
        })
    return results

def simulate_distribution_centers(num_centers=2, seed=200):
    """
    Simulate random distribution centers:
      - id
      - location (lat, lon)
      - demand
      - deadline
    """
    random.seed(seed)
    np.random.seed(seed)

    center_lat, center_lon = 28.8, 77.2
    results = []
    for i in range(num_centers):
        lat_shift = np.random.normal(0, 0.1)
        lon_shift = np.random.normal(0, 0.1)
        demand = random.randint(500, 1500)
        deadline = random.randint(2, 6)

        results.append({
            "id": i,
            "location": [center_lat + lat_shift, center_lon + lon_shift],
            "demand": demand,
            "deadline": deadline
        })
    return results

def simulate_vehicles(num_small=2, num_large=1, seed=300):
    """
    Simulate vehicles:
      - id
      - type ("small" or "large")
      - capacity
      - fixed_cost
      - variable_cost_per_distance
    """
    random.seed(seed)
    results = []
    vid = 0
    # Small
    for _ in range(num_small):
        results.append({
            "id": vid,
            "type": "small",
            "capacity": 1000,
            "fixed_cost": 200,
            "variable_cost_per_distance": 0.8
        })
        vid += 1
    # Large
    for _ in range(num_large):
        results.append({
            "id": vid,
            "type": "large",
            "capacity": 3000,
            "fixed_cost": 350,
            "variable_cost_per_distance": 1.0
        })
        vid += 1

    return results
