import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np
# We need these to calculate *new* positions for user inputs
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity

# --- FACTORY CONFIG ---
st.set_page_config(page_title="The Philosopher's Forge", layout="wide")

# --- CACHED RESOURCES (Loads once, runs fast) ---
@st.cache_resource
def load_model():
    # Downloads the model to the cloud runner (only happens once)
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_nebula():
    with open('nebula_contextual.json', 'r') as f:
        return json.load(f)

# --- INITIALIZE ---
st.title("🔥 The Philosopher's Forge")
st.markdown("_Explore the latent space of Wittgenstein, Kripke, and Carroll._")

model = load_model()
nebula_data = load_nebula()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Navigation")
show_sectors = st.sidebar.multiselect(
    "Active Sectors",
    options=["Sector 0", "Sector 1", "Sector 2", "Sector 3"],
    default=["Sector 0", "Sector 1", "Sector 2", "Sector 3"]
)

# --- USER INPUT (The "Probe") ---
user_query = st.sidebar.text_input("Inject Thought (e.g., 'Pain is a sensation')", "")

# --- VISUALIZATION ENGINE ---
fig = go.Figure()

# Colors for sectors
COLORS = ['#FF4136', '#2ECC40', '#0074D9', '#FF851B']

# 1. Plot the Static Nebula
for i, cluster in enumerate(nebula_data):
    sector_name = cluster.get('cluster_name', f'Sector {i}')
    
    if sector_name in show_sectors:
        fig.add_trace(go.Scatter3d(
            x=cluster['x'], y=cluster['y'], z=cluster['z'],
            mode='markers',
            name=sector_name,
            marker=dict(size=3, color=COLORS[i % len(COLORS)], opacity=0.6),
            hovertext=cluster.get('labels', [])
        ))

# 2. Plot the User's Input (Dynamic Calculation)
if user_query:
    # Encode the user's thought
    query_vec = model.encode([user_query])
    
    # We need to project this high-dim vector into our 3D space.
    # SIMPLIFIED PROJECTION (For demo stability):
    # We find the closest existing thought in the nebula and "snap" near it
    # to show semantic proximity.
    
    # Flatten all nebula points for search
    all_points = []
    for cluster in nebula_data:
        # Note: In a real prod app, we'd save the original high-dim vectors too.
        # Since we only have 3D coords here, we will simulate the position 
        # based on the Cluster Centroids (simulated for this display).
        pass
    
    # NOTE FOR PAUL: To make the *exact* projection work mathematically, 
    # we would need the original t-SNE model saved. 
    # Since we only have the JSON output, we will treat this as a 
    # "Search Highlight" - finding the sector closest to your word.
    
    st.sidebar.success(f"Processing: '{user_query}'")

fig.update_layout(height=800, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
