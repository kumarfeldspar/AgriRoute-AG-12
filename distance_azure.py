
#!/usr/bin/env python
"""
distance_azure.py
-----------------
Generates actual route distances (and polylines) between farms/hubs/centers via Azure Maps if possible.

We store in dist_dict[(t1, id1, t2, id2)] = [
   {
       "route_id": 1,
       "distance_km": X,
       "geometry": [ [lon, lat], [lon, lat], ... ]  # from the polyline
   },
   {
       "route_id": 2,
       "distance_km": X2,
       "geometry": ...
   }
]

If no Azure key or use_azure=False, fallback to Haversine for distance, and geometry is just [start, end].
"""

import math
import requests
import time
import random
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

AZURE_MAPS_KEY = os.getenv("AZURE_MAPS_KEY")

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2)
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def decode_polyline(polyline_str):
    """
    Decodes an Azure polyline (simplified) from the route/directions API.
    Each point is offset-encoded. We can use a standard Google polyline decode library,
    or do a simpler decode if Azure returns geometry in an array. We'll assume standard polyline decode.
    If it doesn't match, you may adapt as needed.

    For demonstration, we handle standard polyline. 
    If you want, you can parse the "legs/points" from the Azure JSON directly.
    """
    # If Azure returns legs/points, we'd just parse that. 
    # Implementation here is left as a placeholder if Azure returns a "polyline" field.
    # For now, we do a stub returning empty or a single line. We'll do a simpler approach:
    return []  # stub

def azure_route(origin, dest):
    """
    Calls Azure Maps to get primary route distance + geometry.
    We do routeRepresentation=polyline, then parse the geometry from "coordinates" in the JSON response.
    Returns (distance_km, geometryList[ [lon,lat], [lon,lat], ... ])
    or (None, None) on failure.
    """
    if not AZURE_MAPS_KEY:
        return None, None
    # origin, dest => [lat, lon]
    base_url = (
        "https://atlas.microsoft.com/route/directions/json"
        f"?api-version=1.0"
        f"&subscription-key={AZURE_MAPS_KEY}"
        f"&query={origin[0]},{origin[1]}:{dest[0]},{dest[1]}"
        "&routeType=shortest"
        "&travelMode=car"
        "&routeRepresentation=polyline"
    )
    resp = requests.get(base_url)
    if resp.status_code == 200:
        data = resp.json()
        try:
            dist_m = data["routes"][0]["summary"]["lengthInMeters"]
            dist_km = dist_m / 1000.0
            # geometry from legs
            # Usually data["routes"][0]["legs"][0]["points"] => array of {latitude,longitude}
            # We'll parse them
            points_data = []
            route_legs = data["routes"][0]["legs"]
            for lg in route_legs:
                for pt in lg["points"]:
                    lat = pt["latitude"]
                    lon = pt["longitude"]
                    points_data.append([lon, lat])
            return dist_km, points_data
        except:
            return None, None
    return None, None

def build_distance_matrix(farms, hubs, centers, use_azure=False):
    """
    Returns dist_dict with up to 2 routes: primary + alt (which we'll just do 1.1x distance).
    For the real route, we either call Azure or do haversine.
    """
    dist_dict = {}

    def get_two_routes(origin, dest):
        # primary
        if use_azure:
            d1, geom1 = azure_route(origin, dest)
            time.sleep(0.1)  # avoid rate-limits
            if d1 is None:
                # fallback
                d1 = haversine_distance(origin[0], origin[1], dest[0], dest[1])
                geom1 = [[origin[1], origin[0]], [dest[1], dest[0]]]  # just line
        else:
            d1 = haversine_distance(origin[0], origin[1], dest[0], dest[1])
            geom1 = [[origin[1], origin[0]], [dest[1], dest[0]]]  # line start->end

        d2 = d1 * 1.1  # alt route
        # We'll do a naive alt geometry shifting midpoints or something
        # but let's keep it simple: same geometry plus a slight offset
        geom2 = geom1[:]  # same route, or we could nudge

        roads = [
            {
                "route_id": 1,
                "distance_km": round(d1, 3),
                "geometry": geom1
            },
            {
                "route_id": 2,
                "distance_km": round(d2, 3),
                "geometry": geom2
            }
        ]
        return roads

    # farm->hub
    for f in farms:
        for h in hubs:
            roads = get_two_routes(f["location"], h["location"])
            dist_dict[("farm", f["id"], "hub", h["id"])] = roads
    # hub->center
    for h in hubs:
        for c in centers:
            roads = get_two_routes(h["location"], c["location"])
            dist_dict[("hub", h["id"], "center", c["id"])] = roads

    return dist_dict
