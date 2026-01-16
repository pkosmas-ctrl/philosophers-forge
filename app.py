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

# --- VISUAL STYLING (The "Sophisticated" Look) ---
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
    Loads the JSON map AND regenerates the vector embeddings 
    so we can do live math in the cloud.
    """
    with open('nebula_contextual.json', 'r') as f:
        data = json.load(f)
    
    # Extract lists for plotting
    df = pd.DataFrame(data)
    
    # We need to re-encode the labels to "hydrate" the brain
    # (We only do the first 500 points to keep startup fast, 
    # or all if you want maximum precision. Let's do a smart sample.)
    
    # SAMPLING STRATEGY: Take top 1000 points to act as "Anchors"
    if len(df) > 1000:
        anchors = df.sample(1000, random_state=42)
    else:
        anchors = df
        
    # Generate vectors for these anchors
    # This is the "Heavy" step that was missing
    anchor_labels = anchors['labels'].apply(lambda x: x[0] if isinstance(x, list) and len(x)>0 else "unknown").tolist()
    anchor_vectors = _model.encode(anchor_labels)
    
    return df, anchors, anchor_vectors

# --- INITIALIZATION ---
st.title("🔥 The Philosopher's Forge")
st.markdown("""
*An interactive exploration of the latent semantic space between **Wittgenstein**, **Kripke**, and **Lewis Carroll**.
Enter a concept to see where it lands in the philosophical nebula.*
""")

# Load Model & Data (with Progress Feedback)
with st.spinner("Igniting the Forge... (Loading AI Model & Re-hydrating Vectors)"):
    model = load_ai_model()
    full_df, anchor_df, anchor_vectors = load_and_hydrate_data(model)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")

# Sector Toggle
all_sectors = sorted(list(set(full_df['cluster_name'].fillna("Unknown"))))
selected_sectors = st.sidebar.multiselect(
    "Visible Sectors",
    all_sectors,
    default=all_sectors
)

# Filter Data based on selection
filtered_df = full_df[full_df['cluster_name'].isin(selected_sectors)]

# --- USER INPUT (The Probe) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Inject New Thought")
user_input = st.sidebar.text_input("", placeholder="e.g. 'Language is a game'")

# --- VISUALIZATION ENGINE ---
fig = go.Figure()

# COLOR PALETTE
COLOR_MAP = {
    'Sector 0': '#FF4B4B', # Red (Consciousness)
    'Sector 1': '#00D4B1', # Cyan (Perception)
    'Sector 2': '#FFC300', # Yellow (Logic)
    'Sector 3': '#7D3C98'  # Purple (Language)
}
DEFAULT_COLOR = '#A6ACAF'

# 1. DRAW THE STATIC NEBULA
for sector in selected_sectors:
    subset = filtered_df[filtered_df['cluster_name'] == sector]
    
    # Get color
    color = COLOR_MAP.get(sector, DEFAULT_COLOR)
    
    fig.add_trace(go.Scatter3d(
        x=subset['x'], y=subset['y'], z=subset['z'],
        mode='markers',
        name=sector,
        marker=dict(size=3, color=color, opacity=0.4),
        hovertext=subset['labels'].apply(lambda l: l[0] if len(l)>0 else ""),
        hoverinfo='text'
    ))

# 2. DRAW THE USER'S THOUGHT (Dynamic Triangulation)
if user_input:
    # A. Calculate Vector
    user_vec = model.encode([user_input])
    
    # B. Find Semantic Neighbors (Cosine Similarity)
    # Compare user thought vs. our 1000 hydrated anchors
    similarities = cosine_similarity(user_vec, anchor_vectors)[0]
    
    # Get Top 5 Indices
    top_5_indices = np.argsort(similarities)[-5:]
    
    # C. Calculate Weighted Position (Barycenter)
    # The new point is pulled towards its neighbors based on similarity strength
    target_x, target_y, target_z = 0, 0, 0
    total_weight = 0
    
    description_lines = []
    
    for idx in top_5_indices:
        # Get the anchor data
        anchor_row = anchor_df.iloc[idx]
        score = similarities[idx]
        
        # Power the score to make strong matches pull harder (Gravity)
        weight = score ** 4 
        
        target_x += anchor_row['x'] * weight
        target_y += anchor_row['y'] * weight
        target_z += anchor_row['z'] * weight
        total_weight += weight
        
        label = anchor_row['labels'][0] if len(anchor_row['labels']) > 0 else "unknown"
        description_lines.append(f"🔗 {label} ({score:.2f})")
        
        # D. Draw "Neural Lines" to these neighbors
        fig.add_trace(go.Scatter3d(
            x=[anchor_row['x'], None], # Placeholder, updated in a moment
            y=[anchor_row['y'], None],
            z=[anchor_row['z'], None],
            mode='lines',
            line=dict(color='white', width=1),
            hoverinfo='none',
            showlegend=False
        ))

    # Final Coordinates
    if total_weight > 0:
        final_x = target_x / total_weight
        final_y = target_y / total_weight
        final_z = target_z / total_weight
        
        # Update lines to connect to the calculated center
        # (Plotly trick: we add individual lines or one connected trace. 
        # For simplicity, let's just add the connection lines now that we know the center)
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

        # Plot the "Injected Thought" Diamond
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
        
        st.sidebar.success(f"Mapped '{user_input}' successfully.")

# --- FINAL LAYOUT ---
fig.update_layout(
    height=800,
    paper_bgcolor='#0E1117', # Matches Streamlit dark mode
    plot_bgcolor='#0E1117',
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor='rgba(0,0,0,0)'
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0,0,0,0.5)",
        font=dict(color="white")
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

st.plotly_chart(fig, use_container_width=True)