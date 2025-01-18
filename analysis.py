import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load data
lp_data = load_json('lp_solution.json')
greedy_data = load_json('greedy_solution.json')
ga_data = load_json('ga_solution.json')

# Extract relevant data for comparison
methods = ['Linear Programming', 'Greedy-KMeans', 'Genetic Algorithm']

data = {
    'Method': methods,
    'Total Cost': [
        lp_data.get('best_cost', 0),
        greedy_data.get('total_cost', 0),
        ga_data.get('best_cost', 0)
    ],
    'Total Spoilage Cost': [
        lp_data.get('total_spoilage_cost', 0),
        sum(cluster.get('cluster_spoilage', 0) for cluster in greedy_data.get('clusters', [])),
        ga_data.get('total_spoilage_cost', 0)
    ]
}

# Create a DataFrame for visualization
df = pd.DataFrame(data)

# Streamlit UI
st.title("Optimization Methods Analytics")
st.write("Analyze and compare the performance of optimization methods based on total cost and spoilage cost.")

# Sidebar filters
st.sidebar.header("Filter Methods")
methods_to_display = st.sidebar.multiselect("Select Methods", methods, default=methods)
filtered_df = df[df['Method'].isin(methods_to_display)]

# Display data table
st.subheader("Comparison Data")
st.dataframe(filtered_df)

# Bar charts for total cost and spoilage cost
st.subheader("Cost Comparisons")
col1, col2 = st.columns(2)

with col1:
    st.write("### Total Cost")
    fig, ax = plt.subplots()
    ax.bar(filtered_df['Method'], filtered_df['Total Cost'], color=['blue', 'green', 'orange'])
    ax.set_xlabel("Method")
    ax.set_ylabel("Total Cost")
    ax.set_title("Total Cost by Method")
    st.pyplot(fig)

with col2:
    st.write("### Total Spoilage Cost")
    fig, ax = plt.subplots()
    ax.bar(filtered_df['Method'], filtered_df['Total Spoilage Cost'], color=['blue', 'green', 'orange'])
    ax.set_xlabel("Method")
    ax.set_ylabel("Total Spoilage Cost")
    ax.set_title("Total Spoilage Cost by Method")
    st.pyplot(fig)

# Pie charts for proportions
st.subheader("Proportional Breakdown")
col1, col2 = st.columns(2)

with col1:
    st.write("### Proportion of Total Cost")
    fig_pie_cost = px.pie(
        filtered_df, values="Total Cost", names="Method",
        title="Total Cost Proportions"
    )
    st.plotly_chart(fig_pie_cost)

with col2:
    st.write("### Proportion of Spoilage Cost")
    fig_pie_spoilage = px.pie(
        filtered_df, values="Total Spoilage Cost", names="Method",
        title="Total Spoilage Cost Proportions"
    )
    st.plotly_chart(fig_pie_spoilage)

# Line chart for trends
st.subheader("Trends Across Methods")
fig_line = px.line(
    filtered_df.melt(id_vars=["Method"], value_vars=["Total Cost", "Total Spoilage Cost"]),
    x="Method", y="value", color="variable",
    title="Trends: Total Cost vs Total Spoilage Cost",
    labels={"value": "Cost", "variable": "Cost Type"}
)
st.plotly_chart(fig_line)

# Scatter plot for cost relationships
st.subheader("Cost Relationships")
fig_scatter = px.scatter(
    filtered_df, x="Total Cost", y="Total Spoilage Cost", color="Method",
    size="Total Spoilage Cost", title="Total Cost vs Total Spoilage Cost",
    labels={"Total Cost": "Total Cost", "Total Spoilage Cost": "Total Spoilage Cost"}
)
st.plotly_chart(fig_scatter)

# Highlight insights
st.subheader("Insights")
min_cost_method = filtered_df.loc[filtered_df['Total Cost'].idxmin(), 'Method']
min_spoilage_method = filtered_df.loc[filtered_df['Total Spoilage Cost'].idxmin(), 'Method']
max_cost_method = filtered_df.loc[filtered_df['Total Cost'].idxmax(), 'Method']
max_spoilage_method = filtered_df.loc[filtered_df['Total Spoilage Cost'].idxmax(), 'Method']

st.write(f"### Lowest Total Cost:")
st.write(f"- *Method:* {min_cost_method}")
st.write(f"- *Cost:* {filtered_df['Total Cost'].min()}")

st.write(f"### Lowest Total Spoilage Cost:")
st.write(f"- *Method:* {min_spoilage_method}")
st.write(f"- *Spoilage Cost:* {filtered_df['Total Spoilage Cost'].min()}")

st.write(f"### Highest Total Cost:")
st.write(f"- *Method:* {max_cost_method}")
st.write(f"- *Cost:* {filtered_df['Total Cost'].max()}")

st.write(f"### Highest Total Spoilage Cost:")
st.write(f"- *Method:* {max_spoilage_method}")
st.write(f"- *Spoilage Cost:* {filtered_df['Total Spoilage Cost'].max()}")

st.write("Use the graphs above to further explore the performance of each method!")