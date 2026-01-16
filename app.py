import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- FACTORY CONFIGURATION ---
st.set_page_config(
    page_title="The Philosopher's Forge",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VISUAL STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stTextInput > label { font-size:1.1rem; color: #4db8ff; font-weight: 500; }
    .stMultiSelect > label { font-size:1.1rem; color: #4db8ff; font-weight: 500; }
    h1 { color: #f0f2f6; }
    p { color: #c4c4c4; }
</style>
""", unsafe_allow_html=True)

# --- 1. THE BRAIN (Cached for Speed) ---
@st.cache_resource
def load_ai_model():
    """Loads the Neural Network (SentenceTransformer)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_and_hydrate_data(_model):
    """
    Loads the JSON map AND 'explodes' it from Clusters into Atoms 
    so we can do math on individual points.
    """
    with open('nebula_contextual.json', 'r') as f:
        data = json.load(f)
    
    # --- FLATTENING PROCESS (The Fix) ---
    flat_points = []
    for sector in data:
        cluster_name = sector.get('cluster_name', 'Unknown')
        # We zip the lists together to create individual points
        # (x[0], y[0], z[0], label[0]) -> Point 0
        if 'x' in sector and 'labels' in sector:
            for x, y, z, lbl in zip(sector['x'], sector['y'], sector['z'], sector['labels']):
                flat_points.append({
                    'cluster_name': cluster_name,
                    'x': x,
                    'y': y,
                    'z': z,
                    'label': lbl # Store as a single string
                })
    
    # Create the DataFrame from the flat atoms
    full_df = pd.DataFrame(flat_points)
    
    # --- SAMPLING (for Speed) ---
    # We take 1000 random points to act as "Anchors" for the brain
    if len(full_df) > 1000:
        anchor_df = full_df.sample(1000, random_state=42)
    else:
        anchor_df = full_df
        
    # --- HYDRATION ---
    # Regenerate vectors only for our anchors
    anchor_vectors = _model.encode(anchor_df['label'].tolist())
    
    return full_df, anchor_df, anchor_vectors

# --- INITIALIZATION ---
st.title("🔥 The Philosopher's Forge")
st.markdown("""
*An interactive exploration of the latent semantic space between **Wittgenstein**, **Kripke**, and **Lewis Carroll**.
Enter a concept to see where it lands in the philosophical nebula.*
""")

# Load Model & Data
with st.spinner("Igniting the Forge... (Unpacking Atoms & Loading Brain)"):
    model = load_ai_model()
    full_df, anchor_df, anchor_vectors = load_and_hydrate_data(model)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")

# Sector Toggle
all_sectors = sorted(list(set(full_df['cluster_name'])))
selected_sectors = st.sidebar.multiselect(
    "Visible Sectors",
    all_sectors,
    default=all_sectors
)

# Filter Data
filtered_df = full_df[full_df['cluster_name'].isin(selected_sectors)]

# --- USER INPUT ---
st.sidebar.markdown("---")
st.sidebar.subheader("Inject New Thought")
user_input = st.sidebar.text_input("", placeholder="e.g. 'Language is a game'")

# --- VISUALIZATION ENGINE ---
fig = go.Figure()

# COLOR PALETTE
COLOR_MAP = {
    'Sector 0': '#FF4B4B', # Red
    'Sector 1': '#00D4B1', # Cyan
    'Sector 2': '#FFC300', # Yellow
    'Sector 3': '#7D3C98'  # Purple
}
DEFAULT_COLOR = '#A6ACAF'

# 1. DRAW THE STATIC NEBULA
for sector in selected_sectors:
    subset = filtered_df[filtered_df['cluster_name'] == sector]
    color = COLOR_MAP.get(sector, DEFAULT_COLOR)
    
    fig.add_trace(go.Scatter3d(
        x=subset['x'], y=subset['y'], z=subset['z'],
        mode='markers',
        name=sector,
        marker=dict(size=3, color=color, opacity=0.4),
        hovertext=subset['label'], # Now it's a simple string column
        hoverinfo='text'
    ))

# 2. DRAW THE USER'S THOUGHT
if user_input:
    # A. Encode Input
    user_vec = model.encode([user_input])
    
    # B. Find Neighbors
    similarities = cosine_similarity(user_vec, anchor_vectors)[0]
    top_5_indices = np.argsort(similarities)[-5:]
    
    # C. Triangulate Position
    target_x, target_y, target_z = 0, 0, 0
    total_weight = 0
    description_lines = []
    
    for idx in top_5_indices:
        anchor_row = anchor_df.iloc[idx] # This is now a SINGLE ROW (Series)
        score = similarities[idx]
        weight = score ** 4 
        
        # MATH IS NOW SAFE (Float * Float)
        target_x += anchor_row['x'] * weight
        target_y += anchor_row['y'] * weight
        target_z += anchor_row['z'] * weight
        total_weight += weight
        
        label = anchor_row['label']
        description_lines.append(f"🔗 {label} ({score:.2f})")
        
        # D. Draw Neural Lines
        fig.add_trace(go.Scatter3d(
            x=[anchor_row['x'], None], # Placeholder
            y=[anchor_row['y'], None],
            z=[anchor_row['z'], None],
            mode='lines',
            line=dict(color='white', width=1),
            hoverinfo='none',
            showlegend=False
        ))

    if total_weight > 0:
        final_x = target_x / total_weight
        final_y = target_y / total_weight
        final_z = target_z / total_weight
        
        # Update lines to connect to center
        for idx in top_5_indices:
            anchor_row = anchor_df.iloc[idx]
            fig.add_trace(go.Scatter3d(
                x=[final_x, anchor_row['x']],
                y=[final_y, anchor_row['y']],
                z=[final_z, anchor_row['z']],
                mode='lines',
                line=dict(color='white', width=2),
                showlegend=False,
                hoverinfo='none'
            ))

        # Plot Diamond
        fig.add_trace(go.Scatter3d(
            x=[final_x], y=[final_y], z=[final_z],
            mode='markers+text',
            name='YOUR THOUGHT',
            text=[user_input],
            textposition="top center",
            textfont=dict(size=14, color='white'),
            marker=dict(size=15, color='#FFFFFF', symbol='diamond'),
            hovertext="<br>".join(description_lines),
            hoverinfo='text'
        ))
        
        st.sidebar.success(f"Mapped '{user_input}'")

# --- FINAL LAYOUT ---
fig.update_layout(
    height=800,
    paper_bgcolor='#0E1117',
    plot_bgcolor='#0E1117',
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='rgba(0,0,0,0)'
    ),
    legend=dict(y=0.99, x=0.01, font=dict(color="white")),
    margin=dict(l=0, r=0, b=0, t=0)
)

st.plotly_chart(fig, use_container_width=True)
