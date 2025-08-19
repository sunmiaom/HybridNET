import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import matplotlib
import os
import ms_entropy as me
import time
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# Configure matplotlib settings
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["font.size"] = 14

# Set output path
output_dir = r"similarities.csv"
os.makedirs(output_dir, exist_ok=True)


# ==================== DATA PROCESSING ====================

# Filter spectra based on precursor m/z difference
def filter_spectra(df, tolerance=0.1):
    unique_mz = set()
    filtered_spectra = []
    for _, row in df.iterrows():
        mz = row["mz"]
        is_unique = True
        for existing_mz in unique_mz:
            if abs(mz - existing_mz) <= tolerance:
                is_unique = False
                break
        if is_unique:
            unique_mz.add(mz)
            filtered_spectra.append({
                "id": row["ID"],
                "precursor_mz": mz,
                "peaks": row["MS2"],
                "type": df.columns.tolist()[0].split("_")[0]
            })
    return filtered_spectra


# Process spectrum string into mz-intensity pairs
def process_spectrum(spectrum_str, clean_spectrum=True):
    if not isinstance(spectrum_str, str):
        print(f"Warning: Input data is not of string type, but {type(spectrum_str)} type")
        return []
    try:
        if not spectrum_str.strip():
            return []
        spectrum = [np.array(item.split(":"), dtype=float) for item in spectrum_str.split(" ") if item.strip()]
        if clean_spectrum:
            return me.clean_spectrum(spectrum, min_ms2_difference_in_da=0.01)
        return spectrum
    except Exception as e:
        print(f"Error processing spectral data: {str(e)}")
        return []


# Load and process the input data
def load_and_process_data():
    print("Loading and processing data...")
    try:
        # Read the data files
        parent_df = pd.read_excel(r"...parent.xlsx")
        product_df = pd.read_excel(r"...product.xlsx")

        # Filter spectra
        parent_spectra = filter_spectra(parent_df)
        product_spectra = filter_spectra(product_df)

        # Process spectra
        parent_spectra = [{
            "id": spec["id"],
            "precursor_mz": spec["precursor_mz"],
            "peaks": process_spectrum(spec["peaks"], clean_spectrum=False),
            "type": "parent"
        } for spec in parent_spectra]

        product_spectra = [{
            "id": spec["id"],
            "precursor_mz": spec["precursor_mz"],
            "peaks": process_spectrum(spec["peaks"]),
            "type": "product"
        } for spec in product_spectra]

        print(
            f"Data processed successfully: Parent ion count {len(parent_spectra)}, Daughter ion count {len(product_spectra)}")
        return parent_spectra, product_spectra, parent_df, product_df

    except Exception as e:
        print(f"Failed to load or process data: {str(e)}")
        exit(1)


# ==================== SIMILARITY CALCULATIONS ====================

# Calculate cosine similarity
def calculate_cosine_similarity(peaks1, peaks2, ms2_tolerance_in_da=0.01):
    if len(peaks1) == 0 or len(peaks2) == 0:
        return 0.0

    peaks1 = np.array(peaks1)
    peaks2 = np.array(peaks2)

    mz_range = np.arange(min(peaks1[:, 0].min(), peaks2[:, 0].min()),
                         max(peaks1[:, 0].max(), peaks2[:, 0].max()) + ms2_tolerance_in_da,
                         ms2_tolerance_in_da)

    vec1 = np.zeros_like(mz_range, dtype=float)
    vec2 = np.zeros_like(mz_range, dtype=float)

    for mz, intensity in peaks1:
        idx = np.abs(mz_range - mz).argmin()
        vec1[idx] = intensity

    for mz, intensity in peaks2:
        idx = np.abs(mz_range - mz).argmin()
        vec2[idx] = intensity

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


# Calculate adjusted cosine similarity
def calculate_adjusted_cosine_similarity(spectrum1, spectrum2, tolerance=0.01):
    if not spectrum1 or not spectrum2:
        return 0.0

    mz_intensity_map1 = {mz: intensity for mz, intensity in spectrum1}
    mz_intensity_map2 = {mz: intensity for mz, intensity in spectrum2}
    all_mz = set(mz_intensity_map1.keys()) | set(mz_intensity_map2.keys())

    vector1 = []
    vector2 = []

    for mz in all_mz:
        matched_intensity1 = 0
        matched_intensity2 = 0

        for mz1, intensity1 in spectrum1:
            if abs(mz1 - mz) <= tolerance:
                matched_intensity1 = max(matched_intensity1, intensity1)

        for mz2, intensity2 in spectrum2:
            if abs(mz2 - mz) <= tolerance:
                matched_intensity2 = max(matched_intensity2, intensity2)

        vector1.append(matched_intensity1)
        vector2.append(matched_intensity2)

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    mean1 = np.mean(vector1)
    mean2 = np.mean(vector2)
    vector1_centered = vector1 - mean1
    vector2_centered = vector2 - mean2

    dot_product = np.dot(vector1_centered, vector2_centered)
    norm1 = np.linalg.norm(vector1_centered)
    norm2 = np.linalg.norm(vector2_centered)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return (similarity + 1) / 2


# Calculate dot product similarity
def calculate_dot_product_similarity(peaks1, peaks2, ms2_tolerance_in_da=0.01):
    if len(peaks1) == 0 or len(peaks2) == 0:
        return 0.0

    peaks1 = np.array(peaks1)
    peaks2 = np.array(peaks2)

    mz_range = np.arange(min(peaks1[:, 0].min(), peaks2[:, 0].min()),
                         max(peaks1[:, 0].max(), peaks2[:, 0].max()) + ms2_tolerance_in_da,
                         ms2_tolerance_in_da)

    vec1 = np.zeros_like(mz_range, dtype=float)
    vec2 = np.zeros_like(mz_range, dtype=float)

    for mz, intensity in peaks1:
        idx = np.abs(mz_range - mz).argmin()
        vec1[idx] = intensity

    for mz, intensity in peaks2:
        idx = np.abs(mz_range - mz).argmin()
        vec2[idx] = intensity

    dot_product = np.dot(vec1, vec2)
    max_possible_dot = np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))

    if max_possible_dot == 0:
        return 0.0
    return dot_product / max_possible_dot


# Calculate neutral loss similarity
def calculate_neutral_loss_similarity(peaks1, peaks2, ms2_tolerance_in_da=0.01):
    if len(peaks1) == 0 or len(peaks2) == 0:
        return 0.0

    peaks1 = np.array(peaks1)
    peaks2 = np.array(peaks2)

    def calculate_neutral_losses(peaks):
        losses = []
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                loss = abs(peaks[i][0] - peaks[j][0])
                intensity = min(peaks[i][1], peaks[j][1])
                losses.append([loss, intensity])
        return np.array(losses) if losses else np.array([])

    losses1 = calculate_neutral_losses(peaks1)
    losses2 = calculate_neutral_losses(peaks2)

    if len(losses1) == 0 or len(losses2) == 0:
        return 0.0

    max_loss = max(losses1[:, 0].max() if len(losses1) > 0 else 0,
                   losses2[:, 0].max() if len(losses2) > 0 else 0)
    loss_range = np.arange(0, max_loss + ms2_tolerance_in_da, ms2_tolerance_in_da)

    vec1 = np.zeros_like(loss_range, dtype=float)
    vec2 = np.zeros_like(loss_range, dtype=float)

    for loss, intensity in losses1:
        idx = np.abs(loss_range - loss).argmin()
        vec1[idx] = intensity

    for loss, intensity in losses2:
        idx = np.abs(loss_range - loss).argmin()
        vec2[idx] = intensity

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


# Calculate entropy similarity
def calculate_entropy_similarity(peaks1, peaks2, ms2_tolerance_in_da=0.01):
    if len(peaks1) == 0 or len(peaks2) == 0:
        return 0.0

    try:
        # Clean the spectra first
        cleaned_peaks1 = me.clean_spectrum(peaks1, min_ms2_difference_in_da=ms2_tolerance_in_da)
        cleaned_peaks2 = me.clean_spectrum(peaks2, min_ms2_difference_in_da=ms2_tolerance_in_da)

        # Calculate entropy similarity
        similarity = me.calculate_entropy_similarity(
            cleaned_peaks1,
            cleaned_peaks2,
            ms2_tolerance_in_da=ms2_tolerance_in_da
        )

        return similarity
    except Exception as e:
        print(f"Error calculating entropy similarity: {str(e)}")
        return 0.0


# ==================== SIMILARITY ANALYSIS PIPELINE ====================

# Run similarity analysis
def run_similarity_analysis(parent_spectra, product_spectra, algorithm, algorithm_name):
    print(f"Running {algorithm_name} similarity analysis...")
    try:
        similarities = []
        total_comparisons = len(parent_spectra) * len(product_spectra)

        with tqdm(total=total_comparisons, desc=f"{algorithm_name} similarity") as pbar:
            for parent in parent_spectra:
                for product in product_spectra:
                    similarity = algorithm(parent["peaks"], product["peaks"], 0.01)
                    if similarity > 0.5:
                        similarities.append({
                            "source": parent["id"],
                            "target": product["id"],
                            "similarity": similarity
                        })
                    pbar.update(1)

        similarities_df = pd.DataFrame(similarities)
        output_file = os.path.join(output_dir, f"{algorithm_name}.csv")
        similarities_df.to_csv(output_file, index=False)
        print(f"Similarity results saved to: {output_file}")
        print(f"Found {len(similarities)} similar spectral pairs")
        return similarities
    except Exception as e:
        print(f"{algorithm_name} similarity calculation failed: {str(e)}")
        return []


# ==================== ALGORITHM EVALUATION ====================

# Load results from all algorithms
def load_algorithm_results(file_paths):
    results = {}
    execution_times = {}
    similarity_matrices = {}
    pairwise_times = {}

    for alg_name, file_path in file_paths.items():
        try:
            start_time = time.time()
            df = pd.read_csv(file_path)
            if 'similarity' not in df.columns:
                print(f"Warning: {alg_name} No 'similarity' column found, skipping")
                continue

            results[alg_name] = df['similarity']

            n = len(df)
            sqrt_n = int(np.sqrt(n))
            if sqrt_n * sqrt_n != n:
                print(f"Warning: {alg_name} Data length {n} not perfect square")
                n = sqrt_n * sqrt_n
                matrix = df['similarity'].values[:n].reshape(sqrt_n, sqrt_n)
            else:
                matrix = df['similarity'].values.reshape(sqrt_n, sqrt_n)

            matrix = (matrix + matrix.T) / 2
            similarity_matrices[alg_name] = matrix

            total_pairs = (sqrt_n * (sqrt_n - 1)) // 2
            execution_time = time.time() - start_time
            avg_pair_time = execution_time / total_pairs
            pairwise_times[alg_name] = avg_pair_time
            execution_times[alg_name] = execution_time

            print(f"Loaded {alg_name} results, data: {len(df)}, time: {execution_time:.2f}s")
            print(f"Mean pairwise time: {avg_pair_time:.6f}s")

        except Exception as e:
            print(f"Failed to load {alg_name} results: {e}")

    return results, execution_times, similarity_matrices, pairwise_times


# Evaluate algorithm performance
def evaluate_algorithms(results, execution_times):
    metrics = {}
    alg_names = list(results.keys())
    similarity_df = pd.DataFrame(results)

    print(f"Shape of clustering input data: {similarity_df.shape}")
    print(f"Missing values:\n{similarity_df.isna().sum()}")
    similarity_df = similarity_df.fillna(similarity_df.mean())

    correlation_matrix = similarity_df.corr()

    for alg in alg_names:
        cv = np.std(results[alg]) / np.mean(results[alg])
        metrics[alg] = {
            'CV': cv,
            'execution_times(s)': execution_times[alg],
            'Correlation': correlation_matrix[alg].drop(alg).mean()
        }

    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(similarity_df)
    silhouette_avg = silhouette_score(similarity_df, cluster_labels)

    for alg in alg_names:
        metrics[alg]['Silhouette'] = silhouette_score(similarity_df[[alg]], cluster_labels)

    return metrics, correlation_matrix, silhouette_avg


# Visualize evaluation results
def visualize_results(metrics, correlation_matrix, silhouette_avg, similarity_matrices, pairwise_times):
    fig = plt.figure(figsize=(16, 12))

    # Correlation heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    annot = np.array([[f"{x:+.2f}" for x in row] for row in correlation_matrix.values])
    sns.heatmap(correlation_matrix, annot=annot, fmt="", cmap='coolwarm', vmin=-1, vmax=1, ax=ax1)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=18, fontweight='bold')

    # Radar chart
    ax2 = fig.add_subplot(2, 2, 2, projection='polar')
    metrics_df = pd.DataFrame(metrics).T[['CV', 'Correlation', 'Silhouette']]
    normalized_metrics = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
    angles = np.linspace(0, 2 * np.pi, len(metrics_df.columns), endpoint=False).tolist()
    angles += angles[:1]

    for i, alg in enumerate(metrics.keys()):
        values = normalized_metrics.loc[alg].values.flatten().tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label=alg)
        ax2.fill(angles, values, alpha=0.1)

    ax2.set_thetagrids(np.degrees(angles[:-1]), metrics_df.columns)
    ax2.set_rlabel_position(45)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=18, fontweight='bold')

    # Parameter sensitivity
    ax3 = fig.add_subplot(2, 2, 3)
    param_values = np.linspace(0.8, 1.2, 5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (alg_name, similarity_matrix) in enumerate(similarity_matrices.items()):
        matrix_differences = []
        for param in param_values:
            modified_matrix = similarity_matrix * param
            diff = np.mean(np.abs(modified_matrix - similarity_matrix))
            matrix_differences.append(diff)
        ax3.plot(param_values, matrix_differences, marker='o', label=alg_name, color=colors[i])

    ax3.set_xlabel('Parameter change', fontsize=14)
    ax3.set_ylabel('Average absolute difference', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=14)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=18, fontweight='bold')

    # Time plot
    ax4 = fig.add_subplot(2, 2, 4)
    bars = ax4.bar(range(len(pairwise_times)), list(pairwise_times.values()), color=colors)
    ax4.set_xticks(range(len(pairwise_times)))
    ax4.set_xticklabels(pairwise_times.keys(), rotation=45, fontsize=14)
    ax4.set_ylabel('Time (μs)', fontsize=14)

    def simple_time_formatter(x, pos):
        if x < 1e-6:
            return f'{x * 1e9:.1f}ns'
        elif x < 1e-3:
            return f'{x * 1e6:.1f}μs'
        elif x < 1:
            return f'{x * 1e3:.1f}ms'
        else:
            return f'{x:.1f}s'

    ax4.yaxis.set_major_formatter(plt.FuncFormatter(simple_time_formatter))
    ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=18, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== MAIN PIPELINE ====================

def main():
    # 1. Load and process data
    parent_spectra, product_spectra, parent_df, product_df = load_and_process_data()

    # 2. Run all similarity analyses
    algorithms = {
        'cosine': calculate_cosine_similarity,
        'adjust-cosine': calculate_adjusted_cosine_similarity,
        'dot_product': calculate_dot_product_similarity,
        'neutral_loss': calculate_neutral_loss_similarity,
        'entropy': me.calculate_entropy_similarity
    }

    all_similarities = {}
    for alg_name, alg_func in algorithms.items():
        similarities = run_similarity_analysis(parent_spectra, product_spectra, alg_func, alg_name)
        all_similarities[alg_name] = similarities

    # 3. Evaluate and compare algorithms
    file_paths = {alg: os.path.join(output_dir, f"{alg}.csv") for alg in algorithms.keys()}
    results, execution_times, similarity_matrices, pairwise_times = load_algorithm_results(file_paths)

    if len(results) >= 2:
        metrics, correlation_matrix, silhouette_avg = evaluate_algorithms(results, execution_times)
        visualize_results(metrics, correlation_matrix, silhouette_avg, similarity_matrices, pairwise_times)

        print("\n" + "=" * 50)
        print("Detailed Evaluation Results:")
        print("=" * 50)
        for alg_name, alg_metrics in metrics.items():
            print(f"\n{alg_name} Results:")
            print(f"Execution time: {alg_metrics['execution_times(s)']:.4f}s")
            print(f"CV: {alg_metrics['CV']:.4f}")
            print(f"Correlation: {alg_metrics['Correlation']:.4f}")
            print(f"Silhouette: {alg_metrics['Silhouette']:.4f}")


if __name__ == "__main__":
    main()