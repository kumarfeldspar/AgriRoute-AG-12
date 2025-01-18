#!/usr/bin/env python
"""
performance_analysis.py
-----------------------
Reads baseline_solution.json and novel_solution.json,
compares cost, spoilage, runtime (if stored), etc.

Outputs:
  comparison.json
"""

import json
import pandas as pd

def main():
    # Read baseline
    with open("baseline_solution.json") as f:
        base = json.load(f)
    # Read novel
    with open("novel_solution.json") as f:
        novel = json.load(f)

    # Build a quick table
    data = []
    data.append({
        "Method": "Baseline",
        "TotalCost": base["total_cost"],
        "Spoilage": base["total_spoilage"],
        "RuntimeSec": base.get("runtime_sec", "N/A")
    })
    data.append({
        "Method": "Novel",
        "TotalCost": round(novel["total_cost"],2),
        "Spoilage": round(novel["total_spoilage"],2),
        "RuntimeSec": novel.get("runtime_sec", "N/A")
    })

    df = pd.DataFrame(data)
    print(df)

    # Save to JSON
    df.to_json("comparison.json", orient="records", indent=4)
    print("Comparison saved: comparison.json")

if __name__ == "__main__":
    main()
