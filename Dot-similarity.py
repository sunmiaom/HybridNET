import numpy as np
import ms_entropy as me
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import matplotlib

matplotlib.use('TkAgg')

# Use the pandas library to read the parent.xlsx and product.xlsx Excel files, storing them in the parent_df and product_df DataFrames respectively.
parent_df = pd.read_excel(r"...parent.xlsx")
product_df = pd.read_excel(r"...product.xlsx")
# Add a dot product similarity calculation function.
def calculate_dot_product_similarity(peaks1, peaks2, ms2_tolerance_in_da=0.01):
    """
    Calculate the dot product similarity between two mass spectra.
    peaks1, peaks2: Mass spectrum peak list, formatted as[mz, intensity]
    ms2_tolerance_in_da: Mass tolerance
    """
    # Check for empty input
    if len(peaks1) == 0 or len(peaks2) == 0:
        return 0.0

    # Convert peaks to numpy arrays
    peaks1 = np.array(peaks1)
    peaks2 = np.array(peaks2)

    # Create vectors.
    mz_range = np.arange(min(peaks1[:, 0].min(), peaks2[:, 0].min()),
                         max(peaks1[:, 0].max(), peaks2[:, 0].max()) + ms2_tolerance_in_da,
                         ms2_tolerance_in_da)

    vec1 = np.zeros_like(mz_range, dtype=float)
    vec2 = np.zeros_like(mz_range, dtype=float)

    # Vector padding
    for mz, intensity in peaks1:
        idx = np.abs(mz_range - mz).argmin()
        vec1[idx] = intensity

    for mz, intensity in peaks2:
        idx = np.abs(mz_range - mz).argmin()
        vec2[idx] = intensity

    # Calculate the dot similarity
    dot_product = np.dot(vec1, vec2)

    # Normalized dot product similarity
    max_possible_dot = np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))
    if max_possible_dot == 0:
        return 0.0

    return dot_product / max_possible_dot


# Helper function to process spectra
def process_spectrum(spectrum_str):
    # Verify if input is of string type
    if not isinstance(spectrum_str, str):
        print(f"Warning: Input data is not of string type, but {type(spectrum_str)} type")
        return []

    try:
        # Process empty string conditions
        if not spectrum_str.strip():
            return []
        spectrum = [np.array(item.split(":"), dtype=float) for item in spectrum_str.split(" ") if item.strip()]
        return me.clean_spectrum(spectrum,
                                 min_ms2_difference_in_da=0.01)
    except Exception as e:
        print(f"Error processing spectral data: {str(e)}")
        return []


# Build parent and product spectral libraries
parent_spectra = [{
    "id": row["ID"],
    "precursor_mz": row["mz"],
    "peaks": process_spectrum(row["MS2"]),
    "type": "parent"
} for _, row in parent_df.iterrows()]

product_spectra = [{
    "id": row["ID"],
    "precursor_mz": row["mz"],
    "peaks": process_spectrum(row["MS2"]),
    "type": "product"
} for _, row in product_df.iterrows()]

# Combine the parent and product spectra libraries
spectral_library_all = parent_spectra + product_spectra

# Calculate similarities between all pairs of parent and product spectra
similarid = set()
for parent in tqdm(parent_spectra):
    for product in product_spectra:
        similarity = calculate_dot_product_similarity(parent["peaks"], product["peaks"], ms2_tolerance_in_da=0.01)
        if similarity > 0.5:
            similarid.add(parent["id"])
            similarid.add(product["id"])

# Create a mapping from ID to index in the spectral library
id_to_index = {spectrum["id"]: idx for idx, spectrum in enumerate(spectral_library_all)}

# Calculate similarities between all pairs of similar IDs
similarities = []
similarid_list = list(similarid)

for i in range(len(similarid_list)):
    for j in range(i + 1, len(similarid_list)):
        idx_i = id_to_index[similarid_list[i]]
        idx_j = id_to_index[similarid_list[j]]
        X = calculate_dot_product_similarity(spectral_library_all[idx_i]["peaks"],
                                             spectral_library_all[idx_j]["peaks"],
                                             ms2_tolerance_in_da=0.01)
        source = spectral_library_all[idx_i]["id"]
        target = spectral_library_all[idx_j]["id"]
        if X > 0.5:
            similarities.append({"source": source, "target": target, "similarity": X})

# Save the similarities to a CSV file
similarities_df = pd.DataFrame(similarities)
similarities_df.to_csv(r"dot.csv",index=False)

# Get all connected nodes based on similarities
connected_nodes = {sim["source"] for sim in similarities} | {sim["target"] for sim in similarities}

# Create the graph and add edges based on similarities
G = nx.Graph()
for sim in similarities:
    if sim["source"] in connected_nodes and sim["target"] in connected_nodes:
        G.add_edge(sim["source"], sim["target"], weight=sim["similarity"])

# Apply Louvain community detection algorithm
partition = community_louvain.best_partition(G)

# Create a color map for each community
community_colors = [partition[node] for node in G.nodes()]

# Define custom colors for communities
custom_colors = ['#B50A2AFF', '#0E84B4FF', '#E48C2AFF', '#574A5EFF', "#14454CFF", "#E75B64FF"]

# Map community numbers to custom colors
node_colors = [custom_colors[community_colors[i] % len(custom_colors)] for i in range(len(community_colors))]

# Get node layout with adjusted spring constant to reduce repulsion between subnetworks
pos = nx.spring_layout(G, k=1.5, iterations=50)

# Map parent and product IDs to m/z values
parent_mz_map = {row["ID"]: row["mz"] for _, row in parent_df.iterrows()}
product_mz_map = {row["ID"]: row["mz"] for _, row in product_df.iterrows()}

# Get the m/z values for connected nodes
node_mz = []
for node in connected_nodes:
    if node in parent_mz_map:
        node_mz.append(parent_mz_map[node])
    elif node in product_mz_map:
        node_mz.append(product_mz_map[node])

node_sizes = [np.log(mz) * 100 for mz in
              node_mz]
# Plot the network with nodes, edges, and labels
edge_weights = [G[u][v]["weight"] * 1.5 for u, v in G.edges]
nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="grey", alpha=0.3)

# Node Drawing,Plot nodes using high-contrast colors
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

# Display partial labels to prevent overlap
# Show labels only for high-degree nodes
degree_threshold = 2
labels = {node: node for node in G.nodes() if G.degree(node) >= degree_threshold}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

# Add title
plt.title('Spectral Similarity Network', pad=20, fontsize=16)

# Adjust layout to ensure full element visibility
plt.tight_layout()

# Save figure
plt.savefig(r'dot.png', dpi=300, bbox_inches='tight')
plt.show()
