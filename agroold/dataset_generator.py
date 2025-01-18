#!/usr/bin/env python
"""
dataset_generator.py
--------------------
Generates synthetic data for:
  - Farms
  - Hubs
  - Centers
  - Vehicles
  - (Optionally) Road disruptions or traffic intensities

Outputs:
  farms.json, hubs.json, centers.json, vehicles.json, disruptions.json
"""

import numpy as np
import pandas as pd
import json
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Adjust these if needed
NUM_FARMS = 10
NUM_HUBS = 3
NUM_CENTERS = 2
NUM_VEHICLES_SMALL = 2
NUM_VEHICLES_LARGE = 2

def generate_farms(num_farms=10, lat0=28.7041, lon0=77.1025):
    """Generate random farm data with location, produce quantity, perishability windows."""
    farms = []
    for i in range(num_farms):
        lat_shift = np.random.normal(0, 0.15)
        lon_shift = np.random.normal(0, 0.15)
        quantity = random.randint(100, 1000)
        perish_window = random.randint(1, 5)  # time units

        farms.append({
            "id": i,
            "location": [lat0 + lat_shift, lon0 + lon_shift],
            "quantity": quantity,
            "perish_window": perish_window
        })
    return farms

def generate_hubs(num_hubs=3, lat0=28.6041, lon0=77.0025):
    """Generate random hub data with capacity, usage cost, location."""
    hubs = []
    for i in range(num_hubs):
        lat_shift = np.random.normal(0, 0.1)
        lon_shift = np.random.normal(0, 0.1)
        capacity = random.randint(2000, 10000)
        usage_cost = random.randint(1000, 5000)

        hubs.append({
            "id": i,
            "location": [lat0 + lat_shift, lon0 + lon_shift],
            "capacity": capacity,
            "usage_cost": usage_cost
        })
    return hubs

def generate_centers(num_centers=2, lat0=28.8041, lon0=77.2025):
    """Generate distribution centers with demand, deadlines, location."""
    centers = []
    for i in range(num_centers):
        lat_shift = np.random.normal(0, 0.1)
        lon_shift = np.random.normal(0, 0.1)
        demand = random.randint(500, 2000)
        deadline = random.randint(2, 7)  # time units

        centers.append({
            "id": i,
            "location": [lat0 + lat_shift, lon0 + lon_shift],
            "demand": demand,
            "deadline": deadline
        })
    return centers

def generate_vehicles(num_small=2, num_large=2):
    """Generate two types of vehicles with different capacities and cost structure."""
    vehicles = []
    vid = 0

    # Small vehicles
    for _ in range(num_small):
        vehicles.append({
            "id": vid,
            "type": "small",
            "capacity": 1500,
            "cost_per_km": 0.6,   # example cost
            "speed": 40          # km/h
        })
        vid += 1

    # Large vehicles
    for _ in range(num_large):
        vehicles.append({
            "id": vid,
            "type": "large",
            "capacity": 3000,
            "cost_per_km": 1.0,
            "speed": 35
        })
        vid += 1

    return vehicles

def generate_disruptions(farms, hubs, centers):
    """
    For dynamic disruptions, we can randomly close roads or add extra cost.
    We'll create a matrix or a simple list describing disruptions between any two points.

    For demonstration, we'll produce a "closed" arc in ~10% of cases.
    Also produce a random "cost_multiplier".
    """
    all_nodes = []
    for f in farms:
        all_nodes.append(("farm", f["id"]))
    for h in hubs:
        all_nodes.append(("hub", h["id"]))
    for c in centers:
        all_nodes.append(("center", c["id"]))

    disruptions = []
    for i in range(len(all_nodes)):
        for j in range(len(all_nodes)):
            if i == j:
                continue
            # random chance to have a closure
            closed = (random.random() < 0.1)  # 10% chance
            # random cost multiplier from 1.0 to 1.5
            cost_mult = round(1.0 + random.random() * 0.5, 2)

            disruptions.append({
                "start": all_nodes[i],
                "end": all_nodes[j],
                "closed": closed,
                "cost_multiplier": cost_mult
            })
    return disruptions

def main():
    farms = generate_farms(NUM_FARMS)
    hubs = generate_hubs(NUM_HUBS)
    centers = generate_centers(NUM_CENTERS)
    vehicles = generate_vehicles(NUM_VEHICLES_SMALL, NUM_VEHICLES_LARGE)
    disruptions = generate_disruptions(farms, hubs, centers)

    with open("farms.json", "w") as f:
        json.dump(farms, f, indent=4)
    with open("hubs.json", "w") as f:
        json.dump(hubs, f, indent=4)
    with open("centers.json", "w") as f:
        json.dump(centers, f, indent=4)
    with open("vehicles.json", "w") as f:
        json.dump(vehicles, f, indent=4)
    with open("disruptions.json", "w") as f:
        json.dump(disruptions, f, indent=4)

    print("Generated farms.json, hubs.json, centers.json, vehicles.json, disruptions.json")

if __name__ == "__main__":
    main()
