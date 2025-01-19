# AgriRoute – Optimized Multi-Stage Agricultural Supply Chain Design

_Team:_ AG12  
_Team Members:_

- Member 1: Piyush Kumar (Team Leader)
- Member 2: Chandra Prakash
- Member 3: Ayush Kumar
- Member 4: Bhaskar
- Member 5: Kanishk Krishnan

---

## Table of Contents

- [AgriRoute – Optimized Multi-Stage Agricultural Supply Chain Design](#agriroute--optimized-multi-stage-agricultural-supply-chain-design)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Problem Statement](#problem-statement)
  - [Solution Approach](#solution-approach)
  - [Features](#features)
  - [Installation](#installation)

---

## Overview

_AgriRoute_ addresses the inefficiencies in agricultural supply chains where perishable goods, limited transport capacities, and dynamic real-world disruptions (e.g., traffic, road closures) challenge traditional logistics. Our project designs a novel, scalable multi-stage optimization algorithm—using a _modified Genetic Algorithm_—that optimizes vehicle routing and resource allocation across farms, storage hubs, and distribution centers, with the objective of minimizing costs and spoilage.

---

## Problem Statement

Modern agricultural supply chains involve several stages:

- _Farms:_ Produce various goods with specific quantities, perishability windows, and geographic locations.
- _Storage Hubs:_ Have limited capacities and incur fixed usage costs.
- _Distribution Centers:_ Demand specific quantities by set deadlines.
- _Fleet:_ Utilizes both small and large vehicles, each with distinct capacities.
- _Constraints:_ Include dynamic factors such as road closures, traffic conditions, and variable fuel costs.

_Input:_

- Data on farms, storage hubs, and distribution centers (locations, capacities, perishability windows, demands).
- Fleet specifications (vehicle sizes and capacities).
- Distance and cost matrices for each route stage.
- Dynamic constraints affecting transportation.

_Output:_

- Optimized routing and vehicle allocation plans.
- Detailed breakdown of total costs and spoilage metrics.
- An analysis of optimal routes and resource allocation to meet deadlines with minimal waste.

---

## Solution Approach

Our innovative solution involves a _modified Genetic Algorithm_, which was tailored to address:

- _Multi-Objective Optimization:_ Balancing total cost, spoilage minimization, and meeting strict delivery deadlines.
- _Dynamic Re-Optimization:_ Adapting to changing conditions such as traffic disruptions and unexpected events.
- _Scalability:_ Handling larger datasets (e.g., 50+ farms, 15+ hubs, 10+ distribution centers) without sacrificing performance.

Additionally, we developed:

- A _simulation and dataset generator_ to create realistic logistics scenarios.
- An _interactive dashboard_ using Streamlit for visualizing routes, vehicle allocations, and performance metrics.

---

## Features

1. _Novel Optimization Algorithm (250 Points)_

   - Implements a modified Genetic Algorithm to solve multi-stage logistical challenges.
   - Balances multiple objectives: cost minimization, spoilage reduction, and deadline compliance.
   - Adapts to dynamic constraints such as traffic and road closures.

2. _Simulation and Dataset Generator (100 Points)_

   - Generate realistic supply chain data including farms, storage hubs, and distribution centers.
   - Customizable parameters for varying simulation complexity, such as perishability rates, vehicle capacities, and demand levels.

3. _Algorithm Benchmarking and Performance Analysis (150 Points)_

   - Compare our algorithm against baseline methods (e.g., greedy heuristics, linear programming).
   - Evaluate key metrics such as total cost, spoilage rate, and computational efficiency.

4. _Interactive Visualization Dashboard (100 Points)_
   - Visualize optimized routes with detailed maps and charts.
   - Display key performance metrics (e.g., total cost, spoilage, time taken) in an intuitive interface.

---

## Installation

Before running the application, ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
Usage
To launch the interactive dashboard, execute the following command:

streamlit run main.py
## Installation

AgriRoute/
├── main.py                    # Main entry-point of the application
├── requirements.txt           # List of Python dependencies
├── src/
│   ├── optimization/          # Contains the modified Genetic Algorithm implementation
│   ├── simulation/            # Dataset generator code for simulating logistics scenarios
│   ├── benchmarking/          # Scripts for performance analysis and benchmarking
│   └── visualization/         # Code for the interactive dashboard
├── data/
│   └── sample_dataset.json    # Example dataset for demonstration
└── README.md                  # Project documentation (this file)


Benchmarking & Performance Analysis
Our benchmarking module provides:

Comparative Charts & Graphs: Visual performance comparisons between our modified Genetic Algorithm and baseline methods.
Detailed Metrics: Analysis of total operational costs, spoilage rates, and computational efficiency.
Scalability Testing: Insights on performance using large-scale datasets with numerous farms, hubs, and distribution centers.
Interactive Visualization Dashboard
The dashboard offers:

Route Visualization: Interactive maps showing optimal routes for both small and large vehicles.
Performance Metrics: Real-time display of key metrics such as cost, spoilage, and processing time.
User-Friendly Interface: Simplified navigation and interactive controls for non-experts.
Contributing
We welcome contributions from the community! To contribute:

Fork this repository.
Create your feature branch (git checkout -b feature/new-feature).
Commit your changes (git commit -am 'Add new feature').
Push to the branch (git push origin feature/new-feature).
Create a new Pull Request.
For major changes, please open an issue first to discuss what you would like to change.
```
