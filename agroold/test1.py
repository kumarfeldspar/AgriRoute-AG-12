import streamlit as st
import numpy as np
import random
import json

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Function to generate random farms
def generate_farms(num_farms, lat0=28.7041, lon0=77.1025):
    farms = []
    for i in range(num_farms):
        lat_shift = np.random.normal(0, 0.15)
        lon_shift = np.random.normal(0, 0.15)
        quantity = random.randint(100, 1000)
        perish_window = random.randint(1, 5)

        farms.append({
            "id": i,
            "location": [lat0 + lat_shift, lon0 + lon_shift],
            "produce_quantity": quantity,
            "perishability_window": perish_window
        })
    return farms

# Function to generate random hubs
def generate_hubs(num_hubs, lat0=28.6041, lon0=77.0025):
    hubs = []
    for i in range(num_hubs):
        lat_shift = np.random.normal(0, 0.1)
        lon_shift = np.random.normal(0, 0.1)
        capacity = random.randint(2000, 10000)
        storage_cost = random.randint(10, 50)
        fixed_usage_cost = random.randint(1000, 5000)

        hubs.append({
            "id": i,
            "location": [lat0 + lat_shift, lon0 + lon_shift],
            "capacity": capacity,
            "storage_cost_per_unit": storage_cost,
            "fixed_usage_cost": fixed_usage_cost
        })
    return hubs

# Function to generate random distribution centers
def generate_centers(num_centers, lat0=28.8041, lon0=77.2025):
    centers = []
    for i in range(num_centers):
        lat_shift = np.random.normal(0, 0.1)
        lon_shift = np.random.normal(0, 0.1)
        demand = random.randint(500, 2000)
        deadline = random.randint(2, 7)

        centers.append({
            "id": i,
            "location": [lat0 + lat_shift, lon0 + lon_shift],
            "demand": demand,
            "deadline": deadline
        })
    return centers

# Function to generate random vehicles
def generate_vehicles(num_small, num_large):
    vehicles = []
    vid = 0

    for _ in range(num_small):
        vehicles.append({
            "id": vid,
            "type": "small",
            "capacity": 1500,
            "fixed_cost": 200,
            "variable_cost_per_distance": 0.6
        })
        vid += 1

    for _ in range(num_large):
        vehicles.append({
            "id": vid,
            "type": "large",
            "capacity": 3000,
            "fixed_cost": 500,
            "variable_cost_per_distance": 1.0
        })
        vid += 1

    return vehicles

# Streamlit app
st.title("Dataset Generator for Logistics Planning")

# Inputs
num_farms = st.sidebar.number_input("Number of Farms", min_value=1, max_value=100, value=10)
num_hubs = st.sidebar.number_input("Number of Hubs", min_value=1, max_value=20, value=3)
num_centers = st.sidebar.number_input("Number of Centers", min_value=1, max_value=10, value=2)
num_small_vehicles = st.sidebar.number_input("Number of Small Vehicles", min_value=1, max_value=10, value=2)
num_large_vehicles = st.sidebar.number_input("Number of Large Vehicles", min_value=1, max_value=10, value=2)

# Manual entry or random generation
mode = st.radio("Choose Input Mode", ["Manual", "Random Generate"])

if mode == "Manual":
    farms = []
    hubs = []
    centers = []

    st.subheader("Enter Farm Details")
    for i in range(num_farms):
        lat = st.number_input(f"Farm {i+1} Latitude", key=f"farm_lat_{i}")
        lon = st.number_input(f"Farm {i+1} Longitude", key=f"farm_lon_{i}")
        quantity = st.number_input(f"Farm {i+1} Produce Quantity", key=f"farm_qty_{i}")
        perish_window = st.number_input(f"Farm {i+1} Perishability Window", key=f"farm_perish_{i}")
        farms.append({"id": i, "location": [lat, lon], "produce_quantity": quantity, "perishability_window": perish_window})

    st.subheader("Enter Hub Details")
    for i in range(num_hubs):
        lat = st.number_input(f"Hub {i+1} Latitude", key=f"hub_lat_{i}")
        lon = st.number_input(f"Hub {i+1} Longitude", key=f"hub_lon_{i}")
        capacity = st.number_input(f"Hub {i+1} Capacity", key=f"hub_capacity_{i}")
        storage_cost = st.number_input(f"Hub {i+1} Storage Cost Per Unit", key=f"hub_cost_{i}")
        fixed_cost = st.number_input(f"Hub {i+1} Fixed Usage Cost", key=f"hub_fixed_{i}")
        hubs.append({"id": i, "location": [lat, lon], "capacity": capacity, "storage_cost_per_unit": storage_cost, "fixed_usage_cost": fixed_cost})

    st.subheader("Enter Center Details")
    for i in range(num_centers):
        lat = st.number_input(f"Center {i+1} Latitude", key=f"center_lat_{i}")
        lon = st.number_input(f"Center {i+1} Longitude", key=f"center_lon_{i}")
        demand = st.number_input(f"Center {i+1} Demand", key=f"center_demand_{i}")
        deadline = st.number_input(f"Center {i+1} Deadline", key=f"center_deadline_{i}")
        centers.append({"id": i, "location": [lat, lon], "demand": demand, "deadline": deadline})

else:
    farms = generate_farms(num_farms)
    hubs = generate_hubs(num_hubs)
    centers = generate_centers(num_centers)

vehicles = generate_vehicles(num_small_vehicles, num_large_vehicles)

# Display generated data
st.subheader("Generated Data")
st.json({"farms": farms, "hubs": hubs, "centers": centers, "vehicles": vehicles})

# Download buttons
def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

if st.button("Download Farms JSON"):
    save_json(farms, "farms.json")
    st.download_button("Download", data=json.dumps(farms, indent=4), file_name="farms.json", mime="application/json")

if st.button("Download Hubs JSON"):
    save_json(hubs, "hubs.json")
    st.download_button("Download", data=json.dumps(hubs, indent=4), file_name="hubs.json", mime="application/json")

if st.button("Download Centers JSON"):
    save_json(centers, "centers.json")
    st.download_button("Download", data=json.dumps(centers, indent=4), file_name="centers.json", mime="application/json")

if st.button("Download Vehicles JSON"):
    save_json(vehicles, "vehicles.json")
    st.download_button("Download", data=json.dumps(vehicles, indent=4), file_name="vehicles.json", mime="application/json")
