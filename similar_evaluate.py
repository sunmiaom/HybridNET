import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
import matplotlib
import time
from scipy.spatial.distance import pdist
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import pearsonr

matplotlib.use('TkAgg')
# Set global font style
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14


def load_algorithm_results(file_paths):
    """load results tables from five algorithms."""
    results = {}
    execution_times = {}
    similarity_matrices = {}
    pairwise_times = {}

    for alg_name, file_path in file_paths.items():
        try:
            start_time = time.time()
            df = pd.read_csv(file_path)
            if 'similarity' not in df.columns:
                print(f"Warning: {alg_name} No 'similarity' column found in results, skipping evaluation")
                continue

            results[alg_name] = df['similarity']

            # Construct a similarity matrix
            n = len(df)
            sqrt_n = int(np.sqrt(n))
            if sqrt_n * sqrt_n != n:
                print(f"Warning: {alg_name} Data length  {n} is not a perfect square, cannot construct square matrix")
                print(f"try construct {sqrt_n}x{sqrt_n} matrices，surplus {n - sqrt_n * sqrt_n} data point")
                n = sqrt_n * sqrt_n
                matrix = df['similarity'].values[:n].reshape(sqrt_n, sqrt_n)
            else:
                matrix = df['similarity'].values.reshape(sqrt_n, sqrt_n)

            # Ensure the matrix is symmetric
            matrix = (matrix + matrix.T) / 2
            similarity_matrices[alg_name] = matrix

            # Calculate the average time for pairwise comparisons.
            total_pairs = (sqrt_n * (sqrt_n - 1)) // 2
            execution_time = time.time() - start_time
            avg_pair_time = execution_time / total_pairs
            pairwise_times[alg_name] = avg_pair_time

            execution_times[alg_name] = execution_time
            print(f"load success  {alg_name} algorithmic result，data: {len(df)}，execution_time: {execution_time:.2f}s")
            print(f"Mean pairwise comparison time: {avg_pair_time:.6f}s")
        except Exception as e:
            print(f"load {alg_name} algorithmic result fail: {e}")
    return results, execution_times, similarity_matrices, pairwise_times


def create_comparison_matrix(results):
    """Create an inter-algorithm similarity comparison matrix"""
    alg_names = list(results.keys())
    n = len(alg_names)
    comparison_matrix = pd.DataFrame(index=alg_names, columns=alg_names)

    #  populate the diagonal (self-similarity scores).
    for i in range(n):
        comparison_matrix.iloc[i, i] = 1.0

    # Calculate inter-algorithm similarity
    for i in range(n):
        for j in range(i + 1, n):
            alg1 = alg_names[i]
            alg2 = alg_names[j]
            corr, _ = stats.pearsonr(results[alg1], results[alg2])
            comparison_matrix.loc[alg1, alg2] = corr
            comparison_matrix.loc[alg2, alg1] = corr

    return comparison_matrix.astype(float)


def evaluate_algorithms(results, execution_times):
    """evaluate the performance of the five algorithms"""
    metrics = {}
    alg_names = list(results.keys())

    # Create a DataFrame with algorithm similarity scores
    similarity_df = pd.DataFrame(results)
    print(f"Shape of clustering input data: {similarity_df.shape}")

    # handle missing values (NaN)
    print(f"check-up missing values：\n{similarity_df.isna().sum()}")
    similarity_df = similarity_df.fillna(similarity_df.mean())

    # 1. compute inter-algorithm correlations
    correlation_matrix = similarity_df.corr()

    # 2. Compute algorithm Variation Coefficient (CV) - Stability Metric
    for alg in alg_names:
        cv = np.std(results[alg]) / np.mean(results[alg])
        metrics[alg] = {
            'CV': cv,
            'execution_times(s)': execution_times[alg]
        }

    # 3. Mean Inter-Algorithm Correlation
    for alg in alg_names:
        avg_corr = correlation_matrix[alg].drop(alg).mean()
        metrics[alg]['Correlation'] = avg_corr

    # 4. Cluster Analysis - Algorithm Consistency
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(similarity_df)
    silhouette_avg = silhouette_score(similarity_df, cluster_labels)

    # 5. Overall Clustering Results
    for alg in alg_names:
        alg_silhouette = silhouette_score(similarity_df[[alg]], cluster_labels)
        metrics[alg]['Silhouette'] = alg_silhouette

    return metrics, correlation_matrix, silhouette_avg


def rank_algorithms(metrics):
    """Rank algorithms based on evaluation metrics"""
    rankings = {}

    # Rank each individual metric
    for metric in metrics[list(metrics.keys())[0]].keys():
        # Define ranking criteria (lower values better for Coefficient of Variation, higher values better for other metrics)
        reverse = False if metric == 'CV' else True
        sorted_algorithms = sorted(
            metrics.items(),
            key=lambda x: x[1][metric],
            reverse=reverse
        )
        rankings[metric] = {alg[0]: i + 1 for i, alg in enumerate(sorted_algorithms)}

    # Calculate composite rankings
    composite_rankings = {}
    for alg in metrics.keys():
        total_rank = 0
        for metric, ranks in rankings.items():
            weight = 2 if metric == 'Composite score' else 1
            total_rank += ranks[alg] * weight
        综合排名[alg] = total_rank / (len(rankings) + 1)

    # then sort by composite score.
    sorted_composite_rankings = sorted(composite_rankings.items(), key=lambda x: x[1])

    return rankings, sorted_composite_rankings

def visualize_results(metrics, correlation_matrix, silhouette_avg, similarity_matrices, pairwise_times):
    """Visualize evaluation results"""
    # Create canvas, specify subplot layout
    fig = plt.figure(figsize=(16, 12))

    # 1. generate inter-algorithm correlation heatmap.
    ax1 = fig.add_subplot(2, 2, 1)
    annot = np.array([[f"{x:+.2f}" for x in row] for row in correlation_matrix.values])
    sns.heatmap(correlation_matrix, annot=annot, fmt="", cmap='coolwarm', vmin=-1, vmax=1, ax=ax1)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=18, fontweight='bold')

    # 2. Radar Chart of Metrics
    ax2 = fig.add_subplot(2, 2, 2, projection='polar')
    alg_names = list(metrics.keys())
    metrics_df = pd.DataFrame(metrics).T

    cols = ['CV', 'Correlation', 'Silhouette']
    metrics_df = metrics_df[cols]

    normalized_metrics = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())

    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]

    for i, alg in enumerate(alg_names):
        values = normalized_metrics.loc[alg].values.flatten().tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label=alg)
        ax2.fill(angles, values, alpha=0.1)

    ax2.set_thetagrids(np.degrees(angles[:-1]), cols)
    ax2.set_rlabel_position(45)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))  # 调整图例位置
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=18, fontweight='bold')

    # 3. Parameter Sensitivity Plot
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

    ax3.set_xlabel('parameter change', fontsize=14)
    ax3.set_ylabel('average absolute difference', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=14)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=18, fontweight='bold')

    # 4. Pairwise Comparison Time Plot
    ax4 = fig.add_subplot(2, 2, 4)
    bars = ax4.bar(range(len(pairwise_times)), list(pairwise_times.values()), color=colors)
    ax4.set_xticks(range(len(pairwise_times)))
    ax4.set_xticklabels(pairwise_times.keys(), rotation=45, fontsize=14)
    ax4.set_ylabel('time (μs)', fontsize=14)

    # Simplify Time Format
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


def main():
    #  Five Algorithm Result Paths
    file_paths = {
        'neutral_loss': r'neutral_loss.csv',
        'entropy': r'entropy.csv',
        'dot_product': r'dot.csv',
        'cosine': r'cosine.csv',
        'adjust-cosine': r'adjust-cosine.csv'
    }

    # Load Algorithm Results
    results, execution_times, similarity_matrices, pairwise_times = load_algorithm_results(file_paths)

    if len(results) < 2:
        print("At least two algorithm results are required for comparison")
        return

    # Evaluate Algorithm Performance
    metrics, correlation_matrix, silhouette_avg = evaluate_algorithms(results, execution_times)

    # Visualize Results
    visualize_results(metrics, correlation_matrix, silhouette_avg, similarity_matrices, pairwise_times)

    #  Print Detailed Assessment
    print("\n" + "=" * 50)
    print("Detailed Assessment Results：")
    print("=" * 50)

    for alg_name, alg_metrics in metrics.items():
        print(f"\n{alg_name} Assessment Results:")
        print(f"execution_times: {alg_metrics['execution_times(s)']:.4f}s")
        print(f"CV: {alg_metrics['CV']:.4f}")
        print(f"Correlation: {alg_metrics['Correlation']:.4f}")
        print(f"Silhouette: {alg_metrics['Silhouette']:.4f}")


if __name__ == "__main__":
    main()
