import os
import psycopg2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import textwrap

load_dotenv()


# Database connection
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

# Fetch data
query = "SELECT id, text, embedding FROM text_embeddings;"
df = pd.read_sql(query, conn)
conn.close()

# Process embeddings
df['embedding'] = df['embedding'].apply(
    lambda x: np.array(eval(x)))  # Convert string to array
vectors = np.vstack(df['embedding'].values)

# Add truncated text for display


def truncate_text(text, max_len=30):
    if len(text) <= max_len:
        return text
    return textwrap.shorten(text, width=max_len, placeholder="...")


df['display_text'] = df['text'].apply(truncate_text)

# Calculate similarity matrix for coloring
similarity_matrix = cosine_similarity(vectors)
avg_similarity = np.mean(similarity_matrix, axis=1)
df['similarity_score'] = avg_similarity

# Apply K-means clustering for better visualization
n_clusters = min(5, len(df))  # Adjust based on data size
if len(df) >= n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(vectors)
else:
    df['cluster'] = 0

# Create 2D PCA projection
pca_2d = PCA(n_components=2)
reduced_vectors_2d = pca_2d.fit_transform(vectors)
df['x'] = reduced_vectors_2d[:, 0]
df['y'] = reduced_vectors_2d[:, 1]

# Create 3D PCA projection
pca_3d = PCA(n_components=3)
reduced_vectors_3d = pca_3d.fit_transform(vectors)
df['z'] = reduced_vectors_3d[:,
                             2] if reduced_vectors_3d.shape[1] > 2 else np.zeros(len(df))

# Create subplot with 2D and 3D visualizations
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "xy"}, {"type": "scene"}]],
    subplot_titles=("2D Vector Space (PCA)", "3D Vector Space (PCA)")
)

# 2D Plot
scatter_2d = px.scatter(
    df, x='x', y='y',
    color='cluster' if 'cluster' in df.columns else None,
    hover_data=['id', 'display_text', 'similarity_score'],
    color_continuous_scale=px.colors.sequential.Viridis,
    size='similarity_score',
    size_max=15,
    opacity=0.8,
)

for trace in scatter_2d.data:
    fig.add_trace(trace, row=1, col=1)

# Add text labels to 2D plot
for i, row in df.iterrows():
    fig.add_annotation(
        x=row['x'],
        y=row['y'],
        text=row['display_text'],
        showarrow=True,
        arrowhead=1,
        row=1, col=1,
        font=dict(size=8),
        ax=0,
        ay=-20,
    )

# 3D Plot
scatter_3d = px.scatter_3d(
    df, x='x', y='y', z='z',
    color='cluster' if 'cluster' in df.columns else None,
    hover_data=['id', 'display_text', 'similarity_score'],
    color_continuous_scale=px.colors.sequential.Viridis,
    size='similarity_score',
    size_max=10,
    opacity=0.8,
)

for trace in scatter_3d.data:
    fig.add_trace(trace, row=1, col=2)

# Update layout
fig.update_layout(
    title_text="Vector Space Visualization of Text Embeddings",
    height=800,
    width=1400,
    showlegend=False,
    template="plotly_white",
)

# Update 2D axes
fig.update_xaxes(title_text="Component 1", row=1, col=1)
fig.update_yaxes(title_text="Component 2", row=1, col=1)

# Update 3D axes
fig.update_scenes(
    xaxis_title="Component 1",
    yaxis_title="Component 2",
    zaxis_title="Component 3",
)

# Add explained variance information
explained_variance_2d = pca_2d.explained_variance_ratio_
explained_variance_3d = pca_3d.explained_variance_ratio_

fig.add_annotation(
    xref="paper", yref="paper",
    x=0.25, y=0,
    text=f"Explained variance: {explained_variance_2d[0]:.2%} (PC1), {
        explained_variance_2d[1]:.2%} (PC2)",
    showarrow=False,
    font=dict(size=10),
)

if len(explained_variance_3d) > 2:
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.75, y=0,
        text=f"Explained variance: {explained_variance_3d[0]:.2%} (PC1), {
            explained_variance_3d[1]:.2%} (PC2), {explained_variance_3d[2]:.2%} (PC3)",
        showarrow=False,
        font=dict(size=10),
    )

# Show the plot
fig.show()

# Save the visualization to HTML file for sharing
fig.write_html("vector_visualization.html")
print("Visualization saved to vector_visualization.html")
