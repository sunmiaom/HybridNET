import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import matplotlib
import os

matplotlib.use('TkAgg')
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# Set the output directory.
output_dir = r"..."
os.makedirs(output_dir, exist_ok=True)

print("Loading data")
try:
    # Read the data files.
    parent_df = pd.read_excel(r"...parent.xlsx")
    product_df = pd.read_excel(r"...product.xlsx")
    print(f"Data loaded successfully：Parent ion count {len(parent_df)}，Daughter ion count {len(product_df)}")
except Exception as e:
    print(f"Failed to load data.: {str(e)}")
    exit(1)


def process_spectrum(spectrum_str):
    if not isinstance(spectrum_str, str):
        print(f"Warning: Input data is not of string type, but {type(spectrum_str)} type")
        return []

    try:
        if not spectrum_str.strip():
            return []

        spectrum = [np.array(item.split(":"), dtype=float) for item in spectrum_str.split(" ") if item.strip()]
        return spectrum
    except Exception as e:
        print(f"Error processing spectral data: {str(e)}")
        return []


def calculate_adjusted_cosine_similarity(spectrum1, spectrum2, tolerance=0.01):
    """
    Calculate the adjusted cosine similarity between two spectra.
    spectrum1, spectrum2: Spectral data, formatted as [[mz1, intensity1], [mz2, intensity2], ...]
    tolerance: Mass-to-charge ratio (m/z) matching tolerance
    """
    if not spectrum1 or not spectrum2:
        return 0.0

    # Create an m/z-to-intensity mapping.
    mz_intensity_map1 = {mz: intensity for mz, intensity in spectrum1}
    mz_intensity_map2 = {mz: intensity for mz, intensity in spectrum2}

    # Get all unique mass-to-charge ratios (m/z).
    all_mz = set(mz_intensity_map1.keys()) | set(mz_intensity_map2.keys())

    # Create vectors.
    vector1 = []
    vector2 = []

    for mz in all_mz:
        # Match peaks within m/z tolerance range
        matched_intensity1 = 0
        matched_intensity2 = 0

        # Identify matching peaks in spectrum1
        for mz1, intensity1 in spectrum1:
            if abs(mz1 - mz) <= tolerance:
                matched_intensity1 = max(matched_intensity1, intensity1)

        # Identify matching peaks in spectrum2
        for mz2, intensity2 in spectrum2:
            if abs(mz2 - mz) <= tolerance:
                matched_intensity2 = max(matched_intensity2, intensity2)

        vector1.append(matched_intensity1)
        vector2.append(matched_intensity2)

    # Convert to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Compute mean values
    mean1 = np.mean(vector1)
    mean2 = np.mean(vector2)

    # Subtract mean values
    vector1_centered = vector1 - mean1
    vector2_centered = vector2 - mean2

    # Calculate the adjusted cosine similarity
    dot_product = np.dot(vector1_centered, vector2_centered)
    norm1 = np.linalg.norm(vector1_centered)
    norm2 = np.linalg.norm(vector2_centered)

    # Return 0 if either vector is a zero vector post-centering
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Similarity calculation
    similarity = dot_product / (norm1 * norm2)

    normalized_similarity = (similarity + 1) / 2

    return normalized_similarity


print("Processing spectral data initiated...")
try:
    # Build a spectral library.
    parent_spectra = [{
        "id": row["ID"],
        "precursor_mz": row["mz"],
        "peaks": process_spectrum(row["MS2"]),
        "type": "parent"
    } for _, row in tqdm(parent_df.iterrows(), desc="Process parent ion spectra")]

    product_spectra = [{
        "id": row["ID"],
        "precursor_mz": row["mz"],
        "peaks": process_spectrum(row["MS2"]),
        "type": "product"
    } for _, row in tqdm(product_df.iterrows(), desc="Process daughter ion spectra")]

    print(f"Data Process successfully：Parent ion count {len(parent_spectra)}，Daughter ion count {len(product_spectra)}")
except Exception as e:
    print(f"Error processing spectral data: {str(e)}")
    exit(1)

print("Similarity calculation...")
try:
    # Similarity calculation
    similarities = []
    total_comparisons = len(parent_spectra) * len(product_spectra)

    with tqdm(total=total_comparisons, desc="Similarity calculation") as pbar:
        for parent in parent_spectra:
            for product in product_spectra:
                similarity = calculate_adjusted_cosine_similarity(parent["peaks"], product["peaks"], tolerance=0.01)
                if similarity > 0.5:  # Save only results with similarity > 0.5
                    similarities.append({
                        "source": parent["id"],
                        "target": product["id"],
                        "similarity": similarity
                    })
                pbar.update(1)

    # Save the similarities to a CSV file
    similarities_df = pd.DataFrame(similarities)
    output_file = os.path.join(output_dir, "adjust-cosine.csv")
    similarities_df.to_csv(output_file, index=False)
    print(f"Similarity results saved to: {output_file}")
    print(f"find {len(similarities)} Similar spectral pairs")
except Exception as e:
    print(f"Similarity calculation failed : {str(e)}")
    exit(1)

print("Create the graph...")
try:
    # Create the graph and add edges based on similarities
    G = nx.Graph()
    for sim in similarities:
        G.add_edge(sim["source"], sim["target"], weight=sim["similarity"])

    partition = community_louvain.best_partition(G)
    community_colors = [partition[node] for node in G.nodes()]

    # Create a color map for each community
    custom_colors = ['#B50A2AFF', '#0E84B4FF', '#E48C2AFF', '#574A5EFF', "#14454CFF", "#E75B64FF"]
    node_colors = [custom_colors[community_colors[i] % len(custom_colors)] for i in range(len(community_colors))]

    # Get node layout with adjusted spring constant to reduce repulsion between subnetworks
    pos = nx.spring_layout(G, k=1.5, iterations=50)

    # Map parent and product IDs to m/z values
    parent_mz_map = {row["ID"]: row["mz"] for _, row in parent_df.iterrows()}
    product_mz_map = {row["ID"]: row["mz"] for _, row in product_df.iterrows()}
    
    node_mz = []
    for node in G.nodes():
        if node in parent_mz_map:
            node_mz.append(parent_mz_map[node])
        elif node in product_mz_map:
            node_mz.append(product_mz_map[node])

    node_sizes = [np.log(mz) * 100 for mz in node_mz]

    # Plot the network with nodes, edges, and labels
    plt.figure(figsize=(12, 10))
    edge_weights = [G[u][v]["weight"] * 1.5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color="grey", alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

    # Display partial labels to prevent overlap
    # Show labels only for high-degree nodes
    degree_threshold = 2
    labels = {node: node for node in G.nodes() if G.degree(node) >= degree_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title('Spectral Similarity Network (Modified Cosine Similarity)', pad=20, fontsize=16)
    plt.tight_layout()

    # Save figure
    output_image = os.path.join(output_dir, "adjust-cosine.png")
    plt.savefig(output_image, dpi=600, bbox_inches='tight')
    print(f"Network graph has been saved to: {output_image}")
    plt.close()
except Exception as e:
    print(f"Failed to construct network graph: {str(e)}")
    exit(1)

print("Processing completed！")