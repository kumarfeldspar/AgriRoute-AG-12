# AgriRoute – Optimized Multi-Stage Agricultural Supply Chain Design

## Team AG12

**Team Members:**

- **Piyush Kumar** (Team Leader)
- **Ayush Kumar**
- **Bhaskar**
- **Chandra Prakash**
- **Kanishk Krishnan**

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Solution Approach](#solution-approach)
4. [Features](#features)
5. [Installation](#installation)
6. [File Structure](#file-structure)
7. [Usage](#usage)
8. [Genetic Algorithm](#genetic-algorithm)

---

## Overview

*AgriRoute* is an innovative solution designed to address inefficiencies in agricultural supply chains. Our system optimizes the movement of perishable goods across farms, storage hubs, and distribution centers by leveraging a **modified Genetic Algorithm** to reduce costs and minimize spoilage. The project integrates advanced algorithms with dynamic constraints to deliver scalable, real-time logistics optimization.

---

## Problem Statement

Modern agricultural supply chains face several challenges:

- **Farms**: Geographic dispersion, varying production capacities, and perishability windows.
- **Storage Hubs**: Limited capacity, high fixed costs, and variable storage costs.
- **Distribution Centers**: Specific demands with strict deadlines.
- **Fleet Management**: Diverse vehicle types with varying capacities and operational costs.
- **Dynamic Constraints**: Traffic, road closures, and fluctuating fuel costs.

![Problem Overview](pdf_images/page_1_img_1.png)

**Inputs:**

- Locations, capacities, and demands for farms, hubs, and distribution centers.
- Fleet specifications and transportation constraints.
- Distance and cost matrices for multi-stage routes.

**Outputs:**

- Optimized routing and vehicle allocation plans.
- Detailed cost and spoilage analyses.
- Visualization of routes and key performance metrics.

---

## Solution Approach

Our solution utilizes a **modified Genetic Algorithm** to:

1. Optimize multi-stage supply chains with competing objectives:
   - Minimize costs.
   - Reduce spoilage.
   - Meet delivery deadlines.
2. Handle dynamic real-world disruptions such as traffic and road closures.
3. Scale efficiently to support large datasets (e.g., 50+ farms, 20+ hubs, 15+ distribution centers).

![Solution Overview](pdf_images/page_2_img_1.png)

### Highlights

- **Simulation Tools**: Generate realistic logistics scenarios.
- **Interactive Dashboard**: Visualize routes, allocations, and metrics.
- **Benchmarking**: Compare against baseline methods like Greedy Heuristics and Linear Programming.

> **Note:** This project also utilizes IBM's CPLEX optimization software, a community platform free for educational institutions. Ensure that the education version of CPLEX is installed to execute the relevant Python files.

---

## Features

### 1. **Optimization Algorithm**

- Implements a **modified Genetic Algorithm** tailored for multi-stage logistical challenges.
- Balances cost, spoilage, and deadline compliance.
- Adapts to dynamic constraints.

### 2. **Simulation and Dataset Generator**

- Generate realistic data for farms, hubs, and centers.
- Customize parameters like perishability rates and vehicle capacities.

### 3. **Benchmarking and Performance Analysis**

- Compare our algorithm with baseline methods.
- Evaluate metrics like total cost, spoilage, and computational efficiency.

### 4. **Interactive Visualization Dashboard**

- Visualize optimized routes on detailed maps.
- Display real-time metrics such as cost and spoilage.
- Simplified user interface for non-experts.

![Dashboard Overview](pdf_images/page_3_img_1.png)

---

## Installation

Follow these steps to set up and run the application:

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd AgriRoute
   ```

2. **Set Up Environment Variables**

   - Rename `.env.example` to `.env`:
     ```bash
     mv .env.example .env
     ```
   - Generate an **Azure Maps API Key**:
     1. Create a free account on Azure.
     2. Create a Maps resource.
     3. Navigate to the **Authentication** section in the Azure portal.
     4. Copy the **Primary Key** and paste it into the `.env` file.

3. **Install IBM CPLEX**

   - Download and install the educational version of IBM CPLEX Optimization Studio from [IBM Academic Initiative](https://www.ibm.com/academic).
   - Ensure the CPLEX Python API is correctly installed.

4. **Create a Virtual Environment**

   - Using Python 3.10 (recommended):
     ```bash
     python3.10 -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     ```

5. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

6. **Run the Application**

   ```bash
   streamlit run main.py
   ```

---

## File Structure

```plaintext
AgriRoute/
├── .env.example               # Environment variable template
├── main.py                    # Entry point for the Streamlit application
├── requirements.txt           # Python dependencies
├── data_input.py              # Simulation and data generation
├── distance_azure.py          # Distance matrix calculation
├── ga_mutation.cpp            # Genetic Algorithm core (C++ implementation)
├── ga_mutation.py             # Genetic Algorithm wrapper (Python)
├── linear_programming.py      # Linear Programming benchmark
├── greedy_kmeans.py           # Greedy heuristic benchmark
├── map_data.json              # Sample map data
├── ga_solution.json           # Genetic Algorithm solution output
├── Readme.md                  # Documentation (this file)
└── comparison.json            # Benchmarking results
```

---

## Usage

### Launch the Dashboard

Run the following command to start the interactive dashboard:

```bash
streamlit run main.py
```

### Explore Features

- Input simulated or manual logistics data (farms, hubs, centers).
- Generate distance matrices using Azure Maps.
- Run optimization methods (Genetic Algorithm, Linear Programming, Greedy Heuristics).
- Visualize routes and performance metrics in real-time.

---

## Genetic Algorithm

The **Genetic Algorithm (GA)** is a bio-inspired optimization technique that mimics the process of natural selection. In *AgriRoute*, GA is tailored to:

1. **Chromosome Representation:**

   - Each chromosome represents a route plan (vehicle assignments and paths).

2. **Fitness Function:**

   - Evaluates total cost, spoilage, and deadline compliance.

3. **Genetic Operators:**

   - **Selection:** Chooses the best chromosomes based on fitness.
   - **Crossover:** Combines two chromosomes to generate offspring.
   - **Mutation:** Introduces randomness to explore new solutions.

4. **Advantages:**

   - Handles multi-objective optimization.
   - Adapts to dynamic constraints (e.g., traffic disruptions).

---

Thank you for exploring AgriRoute! We hope this project inspires innovative solutions for agricultural supply chain optimization. For more details, refer to the attached [AG-12 PPT.pdf](AG-12%20PPT.pdf).

