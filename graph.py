import streamlit as st
import json
import folium
from streamlit_folium import st_folium

def load_data():
    """
    Loads JSON data from 'map_data.json' and 'ga_solution.json'
    located in the same directory.
    """
    with open('map_data.json', 'r') as f:
        map_data = json.load(f)

    with open('ga_solution.json', 'r') as f:
        ga_solution = json.load(f)

    return map_data, ga_solution

def create_map(map_data, ga_solution):
    """
    Creates a Folium map showing:
    - Markers for farms, hubs, and centers with labels.
    - Polylines for each route in ga_solution: farm -> hub -> center.
    """

    # ----------------------------------------------------------------
    # 1. Prepare a base map
    #    We'll pick a center location somewhat in the middle of the data.
    #    You can adjust the center/zoom as needed.
    # ----------------------------------------------------------------
    initial_coords = [28.7, 77.1]  # approximate center around Delhi
    folium_map = folium.Map(location=initial_coords, zoom_start=10)

    # ----------------------------------------------------------------
    # 2. Extract farm, hub, center info from map_data
    # ----------------------------------------------------------------
    farms = map_data['farms']
    hubs = map_data['hubs']
    centers = map_data['centers']

    # Convert them to dictionaries keyed by their 'id' for quick lookup
    farm_dict = {f['id']: f for f in farms}
    hub_dict = {h['id']: h for h in hubs}
    center_dict = {c['id']: c for c in centers}

    # ----------------------------------------------------------------
    # 3. Add markers for farms, hubs, centers
    # ----------------------------------------------------------------
    # Farm markers
    for f in farms:
        f_lat, f_lng = f['location'][0], f['location'][1]
        popup_text = (
            f"<b>Farm #{f['id']}</b><br>"
            f"Produce Quantity: {f['produce_quantity']}<br>"
            f"Perishability: {f['perishability_window']} day(s)"
        )
        folium.Marker(
            location=[f_lat, f_lng],
            popup=popup_text,
            tooltip=f"Farm #{f['id']}",
            icon=folium.Icon(color="green", icon="leaf", prefix="fa")
        ).add_to(folium_map)

    # Hub markers
    for h in hubs:
        h_lat, h_lng = h['location'][0], h['location'][1]
        popup_text = (
            f"<b>Hub #{h['id']}</b><br>"
            f"Capacity: {h['capacity']}<br>"
            f"Storage Cost/Unit: {h['storage_cost_per_unit']}<br>"
            f"Fixed Usage Cost: {h['fixed_usage_cost']}"
        )
        folium.Marker(
            location=[h_lat, h_lng],
            popup=popup_text,
            tooltip=f"Hub #{h['id']}",
            icon=folium.Icon(color="blue", icon="warehouse", prefix="fa")
        ).add_to(folium_map)

    # Center markers
    for c in centers:
        c_lat, c_lng = c['location'][0], c['location'][1]
        popup_text = (
            f"<b>Center #{c['id']}</b><br>"
            f"Demand: {c['demand']}<br>"
            f"Deadline: {c['deadline']} day(s)"
        )
        folium.Marker(
            location=[c_lat, c_lng],
            popup=popup_text,
            tooltip=f"Center #{c['id']}",
            icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa")
        ).add_to(folium_map)

    # ----------------------------------------------------------------
    # 4. Plot routes from ga_solution
    #
    #    For each route:
    #    - We have farm_id, hub_id, center_id in route dictionary.
    #    - We'll create a polyline farm->hub->center
    #    - We'll color the route if it's spoiled or not.
    #    - We'll add a popup with route details: cost, time, etc.
    # ----------------------------------------------------------------
    detailed_routes = ga_solution.get('detailed_routes', [])

    for route in detailed_routes:
        farm_id = route['farm_id']
        hub_id = route['hub_id']
        center_id = route['center_id']

        # Get lat-lons
        f_lat, f_lng = farm_dict[farm_id]['location']
        h_lat, h_lng = hub_dict[hub_id]['location']
        c_lat, c_lng = center_dict[center_id]['location']

        # Choose a color to indicate spoilage or not
        # For example: Red if 'spoiled' is True, else green
        route_color = "red" if route["spoiled"] else "green"

        # Create a list of points for the polyline
        route_points = [
            (f_lat, f_lng),
            (h_lat, h_lng),
            (c_lat, c_lng)
        ]

        # Construct popup or tooltip with route info
        popup_info = (
            f"<b>Route:</b> Farm #{farm_id} → Hub #{hub_id} → Center #{center_id}<br>"
            f"Vehicle ID: {route['vehicle_id']}<br>"
            f"Quantity: {route['quantity']}<br>"
            f"Dist (Farm-Hub): {route['dist_farm_hub']} km<br>"
            f"Dist (Hub-Center): {route['dist_hub_center']} km<br>"
            f"Total Dist: {route['total_dist']} km<br>"
            f"Total Time: {route['total_time']}<br>"
            f"Route Cost: {route['route_cost']}<br>"
            f"On Time: {route['on_time']}<br>"
            f"Spoiled: {route['spoiled']}"
        )

        # Add polyline to map
        folium.PolyLine(
            locations=route_points,
            color=route_color,
            weight=4,
            tooltip="Click for route info",
            popup=popup_info
        ).add_to(folium_map)

    return folium_map

def main():
    st.set_page_config(page_title="Logistics Routes", layout="wide")
    st.title("Logistics Visualization")

    # Load data
    map_data, ga_solution = load_data()

    # Create the Folium map
    folium_map = create_map(map_data, ga_solution)

    # Display in Streamlit
    st_folium(
        folium_map,
        width=1000,
        height=600
    )

if __name__ == "__main__":
    main()
