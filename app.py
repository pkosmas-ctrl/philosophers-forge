import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- FACTORY CONFIG ---
st.set_page_config(page_title="The Philosopher's Forge", layout="wide")

# --- CUSTOM CSS FOR "SOPHISTICATED" LOOK ---
st.markdown("""
<style>
    .stTextInput > label {font-size:1.2rem; font-weight:bold; color:#FF4B4B;}
    div[data-testid="stSidebar"] {background-color: #111;}
</style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCES ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_nebula():
    with open('nebula_contextual.json', 'r') as f:
        data = json.load(f)
        # Pre-compute vectors for fast searching if they aren't saved
        # (Since we only saved 3D coords, we can't do exact vector search on the past 
        # unless we re-encode. For speed, we will assume the textual labels are enough 
        # or we just rely on the 3D proximity if we had the vectors. 
        # BETTER APPROACH: We will Re-Encode the top 100 anchor terms on startup 
        # to create a "Search Grid".)
        return data

# --- INITIALIZE ---
st.title("🔥 The Philosopher's Forge: V2")
st.markdown("_Latent Space Explorer: Wittgenstein, Kripke, Carroll_")

model = load_model()
nebula_data = load_nebula()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Navigation")
show_sectors = st.sidebar.multiselect(
    "Active Sectors",
    options=["Sector 0", "Sector 1", "Sector 2", "Sector 3"],
    default=["Sector 0", "Sector 1", "Sector 2", "Sector 3"]
)

# --- USER INPUT ---
user_query = st.sidebar.text_input("Inject Thought", placeholder="e.g., 'Pain is private'")

# --- VISUALIZATION ENGINE ---
fig = go.Figure()

# Colors
COLORS = ['#FF4136', '#2ECC40', '#0074D9', '#FF851B']
label_list = []
x_list, y_list, z_list = [], [], []

# 1. Build the Static Universe
for i, cluster in enumerate(nebula_data):
    sector_name = cluster.get('cluster_name', f'Sector {i}')
    
    if sector_name in show_sectors:
        # Save points for "Anchor" calculation
        x_list.extend(cluster['x'])
        y_list.extend(cluster['y'])
        z_list.extend(cluster['z'])
        label_list.extend(cluster.get('labels', []))

        fig.add_trace(go.Scatter3d(
            x=cluster['x'], y=cluster['y'], z=cluster['z'],
            mode='markers',
            name=sector_name,
            marker=dict(size=3, color=COLORS[i % len(COLORS)], opacity=0.5),
            hovertext=cluster.get('labels', []),
            hoverinfo='text'
        ))

# 2. THE NEW ENGINE: Semantic Triangulation
if user_query:
    with st.spinner(f"Triangulating '{user_query}' in latent space..."):
        # A. Encode User Input
        user_vec = model.encode([user_query])
        
        # B. Find Nearest Neighbors (We have to re-encode a subset of labels to match)
        # Optimization: We pick 200 random labels from the active sectors to compare against
        # (Real-time encoding of 3000 points is too slow, so we sample)
        sample_indices = np.linspace(0, len(label_list)-1, 150, dtype=int)
        sample_labels = [label_list[i] for i in sample_indices]
        
        # Encode the anchors
        anchor_vecs = model.encode(sample_labels)
        
        # Calculate Similarity
        sims = cosine_similarity(user_vec, anchor_vecs)[0]
        
        # Get Top 3 matches
        top_indices = np.argsort(sims)[-3:] # Top 3
        
        # Calculate Weighted Position (Triangulation)
        # We take the 3D coordinates of the Top 3 matches and average them
        target_x, target_y, target_z = 0, 0, 0
        total_weight = 0
        
        description_text = "<b>Connected Concepts:</b><br>"
        
        for idx in top_indices:
            real_idx = sample_indices[idx]
            match_label = label_list[real_idx]
            weight = sims[idx] ** 2 # Square it to pull harder towards strong matches
            
            target_x += x_list[real_idx] * weight
            target_y += y_list[real_idx] * weight
            target_z += z_list[real_idx] * weight
            total_weight += weight
            
            description_text += f"• {match_label} ({sims[idx]:.2f})<br>"
            
            # Draw a "Neural Line" from the new point to this anchor
            fig.add_trace(go.Scatter3d(
                x=[x_list[real_idx], target_x/total_weight], 
                y=[y_list[real_idx], target_y/total_weight], 
                z=[z_list[real_idx], target_z/total_weight],
                mode='lines',
                line=dict(color='white', width=2),
                showlegend=False
            ))

        # Final Coordinates
        final_x = target_x / total_weight
        final_y = target_y / total_weight
        final_z = target_z / total_weight

        # Plot the Injected Thought (The Glowing Orb)
        fig.add_trace(go.Scatter3d(
            x=[final_x], y=[final_y], z=[final_z],
            mode='markers+text',
            name='INJECTED THOUGHT',
            text=[user_query],
            textposition="top center",
            marker=dict(size=12, color='#FFFFFF', symbol='diamond', opacity=1.0),
            hovertext=description_text,
            hoverinfo='text'
        ))
        
        st.sidebar.success("Triangulation Complete.")

# 3. Layout Polish
fig.update_layout(
    height=800, 
    template="plotly_dark",
    scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        bgcolor='rgba(0,0,0,0)' # Transparent background
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

st.plotly_chart(fig, use_container_width=True)