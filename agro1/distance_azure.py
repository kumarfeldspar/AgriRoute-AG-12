#!/usr/bin/env python
"""
distance_azure.py
-----------------
Generates distance & cost matrix among farms, storage hubs, and distribution centers.

If Azure Maps API is available, we can request actual distances, toll costs, etc.
Otherwise, fallback to a simple Euclidean approach.

Usage:
  1) Provide an Azure Maps key if available.
  2) Provide lists of farms, hubs, centers.
  3) We produce a nested dictionary or a DataFrame with distances/cost multipliers.
"""

import math
import requests
import time

# Optionally set your Azure Maps key here
AZURE_MAPS_KEY = "4j0DlBhKmXzbG2frDTurBFfAq6NPysfILisACKoao83EcEjvmEobJQQJ99BAACYeBjFeh7q1AAAgAZMP3Pj2"  # e.g., "YOUR_AZURE_MAPS_KEY"

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns distance in kilometers using the Haversine formula.
    """
    R = 6371.0  # Earth radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def build_distance_matrix(farms, hubs, centers, use_azure=False):
    """
    Returns a dictionary: dist[('farm', fid, 'hub', hid)] = distance_in_km
    and similarly for hub->center, farm->center if needed.

    If use_azure=True and AZURE_MAPS_KEY is set, attempts actual route distance.
    Otherwise uses Haversine as fallback.
    """
    dist_dict = {}

    # Utility function to get Azure route distance (single origin->destination)
    def get_azure_route_distance(origin, dest):
        # origin, dest => [lat, lon]
        base_url = ("https://atlas.microsoft.com/route/directions/json"
                    f"?subscription-key={AZURE_MAPS_KEY}"
                    "&api-version=1.0"
                    f"&query={origin[0]},{origin[1]}:{dest[0]},{dest[1]}"
                    "&routeType=shortest&travelMode=car")
        resp = requests.get(base_url)
        if resp.status_code == 200:
            data = resp.json()
            try:
                dist_meters = data["routes"][0]["summary"]["lengthInMeters"]
                return dist_meters / 1000.0  # convert to km
            except:
                return None
        else:
            return None

    # Farm->Hub
    for f in farms:
        for h in hubs:
            o = f["location"]
            d = h["location"]
            if use_azure and AZURE_MAPS_KEY:
                dist_km = get_azure_route_distance(o, d)
                time.sleep(0.05)  # avoid hitting rate limits
                if dist_km is None:
                    dist_km = haversine_distance(o[0], o[1], d[0], d[1])
            else:
                dist_km = haversine_distance(o[0], o[1], d[0], d[1])

            dist_dict[("farm", f["id"], "hub", h["id"])] = dist_km

    # Hub->Center
    for h in hubs:
        for c in centers:
            o = h["location"]
            d = c["location"]
            if use_azure and AZURE_MAPS_KEY:
                dist_km = get_azure_route_distance(o, d)
                time.sleep(0.05)
                if dist_km is None:
                    dist_km = haversine_distance(o[0], o[1], d[0], d[1])
            else:
                dist_km = haversine_distance(o[0], o[1], d[0], d[1])

            dist_dict[("hub", h["id"], "center", c["id"])] = dist_km

    # If needed, farm->center direct or other combos can be added similarly.

    return dist_dict
