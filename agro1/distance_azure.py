
#!/usr/bin/env python
"""
distance_azure.py
-----------------
Generates possible road distances among farms, storage hubs, and distribution centers.

- If AZURE_MAPS_KEY is set and use_azure=True, attempts real distances from Azure.
- Otherwise uses Haversine or radial approximation.

We store multiple possible roads per pair (2 max) for demonstration.
"""

import math
import requests
import time
import random

# Optionally set your Azure Maps key here
AZURE_MAPS_KEY = "4j0DlBhKmXzbG2frDTurBFfAq6NPysfILisACKoao83EcEjvmEobJQQJ99BAACYeBjFeh7q1AAAgAZMP3Pj2"  # e.g. "YOUR_REAL_AZURE_KEY"

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
    Returns a dict:
        dist_dict[(type1, id1, type2, id2)] = [
           { "route_id": 1, "distance_km": X1 },
           { "route_id": 2, "distance_km": X2 },
           ...
        ]
    We do farm->hub and hub->center pairs. (Farm->center can be added similarly if needed.)
    """
    dist_dict = {}

    def azure_route_distance(origin, dest):
        """
        Returns route distance in km if successful, else None.
        """
        if not AZURE_MAPS_KEY:
            return None
        base_url = (
            "https://atlas.microsoft.com/route/directions/json"
            f"?subscription-key={AZURE_MAPS_KEY}"
            "&api-version=1.0"
            f"&query={origin[0]},{origin[1]}:{dest[0]},{dest[1]}"
            "&routeType=shortest&travelMode=car"
        )
        resp = requests.get(base_url)
        if resp.status_code == 200:
            data = resp.json()
            try:
                dist_meters = data["routes"][0]["summary"]["lengthInMeters"]
                return dist_meters / 1000.0  # convert to km
            except:
                return None
        return None

    def compute_all_roads(origin, dest):
        """
        Returns up to 2 roads (primary + secondary).
        """
        if use_azure:
            d1 = azure_route_distance(origin, dest)
            time.sleep(0.05)
            if d1 is None:
                # fallback
                d1 = haversine_distance(origin[0], origin[1], dest[0], dest[1])
        else:
            d1 = haversine_distance(origin[0], origin[1], dest[0], dest[1])

        if d1 is None:
            return []
        d2 = round(d1 * 1.1, 3)
        roads = [
            {"route_id": 1, "distance_km": round(d1, 3)},
            {"route_id": 2, "distance_km": d2}
        ]
        return roads

    # Farm->Hub
    for f in farms:
        for h in hubs:
            roads = compute_all_roads(f["location"], h["location"])
            dist_dict[("farm", f["id"], "hub", h["id"])] = roads

    # Hub->Center
    for h in hubs:
        for c in centers:
            roads = compute_all_roads(h["location"], c["location"])
            dist_dict[("hub", h["id"], "center", c["id"])] = roads

    return dist_dict
