import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
from typing import Dict, List


def plot_adjacency_matrix(adj_matrix, nodes):
    """
    Show adjacency matrix as a heatmap with node labels on x/y axes.
    """
    plt.figure(figsize=(18, 16))
    plt.imshow(adj_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label='Edge Weight')
    plt.xticks(range(len(nodes)), nodes, rotation=90)
    plt.yticks(range(len(nodes)), nodes)
    plt.title("Adjacency Matrix Heatmap")
    plt.show()


def plot_graph_partition(G, partition_dict, pos=None, title="Graph Partition"):
    """
    Plots a partitioned NetworkX graph.
    `partition_dict` is { cluster_label: [node_ids], ... }
    If you have a precomputed layout (pos), pass it in; otherwise uses spring_layout.
    """
    if pos is None:
        pos = nx.spring_layout(G)
    colors = ["lightblue", "salmon", "palegreen", "orange", "violet", "yellow"]
    
    plt.figure(figsize=(18,16))
    
    for i, (c_idx, members) in enumerate(partition_dict.items()):
        color = colors[i % len(colors)]
        nx.draw_networkx_nodes(G, pos, nodelist=members, node_color=color, label=f"Cluster {c_idx}")
        
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(title)
    plt.legend()
    plt.show()
    

def plot_bar_comparison(labels, values, ylabel="", title="Bar Comparison"):
    """
    Plots a single bar chart with the given labels on x-axis and values as heights.
    """
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_grouped_bar_comparison(labels_methods, data_list, data_labels, title="Grouped Bar Comparison", width=0.25):
    """
    data_list: list of lists, e.g. [intra_pairwise_vals, intra_centroid_vals, across_vals]
    data_labels: e.g. ["Intra Pairwise Dist", "Intra Centroid Dist", "Across Dist"]
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels_methods))
    
    # We want to shift each bar group
    n_bars = len(data_list)
    total_width = width * n_bars
    offset_start = -total_width / 2.0

    for i, (dvals, dlabel) in enumerate(zip(data_list, data_labels)):
        offset = offset_start + i * width
        plt.bar(x + offset, dvals, width, label=dlabel)

    plt.xticks(x, labels_methods)
    plt.ylabel("Distance (1 - cos. similarity)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dendrogram(linkage_matrix, labels):
    """
    Plots a dendrogram for hierarchical clustering.
    """
    plt.figure(figsize=(16, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering Dendrogram (Ward)')
    plt.xlabel('Nodes')
    plt.ylabel('Cosine Distance')
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(results_df: pd.DataFrame, title: str = "Community Detection Comparison") -> Dict[str, int]:
    """
    Create comprehensive visualization of evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        title: Title for the plots
        
    Returns:
        Dictionary with ranking summary
    """
    if len(results_df) == 0:
        print("No results to plot")
        return {}
    
    # Create subplots for key metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Key metrics to visualize
    metrics_to_plot = [
        ('modularity', 'Modularity (Higher is Better)'),
        ('coverage', 'Coverage (Higher is Better)'),
        ('performance', 'Performance (Higher is Better)'),
        ('silhouette_score', 'Silhouette Score (Higher is Better)'),
        ('conductance', 'Conductance (Lower is Better)'),
        ('calinski_harabasz_score', 'Calinski-Harabasz Score (Higher is Better)')
    ]
    
    # Colors for each method
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    for i, (metric, metric_title) in enumerate(metrics_to_plot):
        if metric not in results_df.columns:
            continue
            
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Create bar plot
        bars = ax.bar(results_df['method'], results_df[metric], 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize plot
        ax.set_title(metric_title, fontweight='bold', fontsize=10)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Add value labels on bars
        for bar, value in zip(bars, results_df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Highlight best performer (red border)
        if metric in ['conductance']:  # Lower is better
            best_idx = results_df[metric].idxmin()
        else:  # Higher is better
            best_idx = results_df[metric].idxmax()
        
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
    
    # Remove unused subplot if needed
    if len(metrics_to_plot) < 6:
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate composite ranking based on key metrics
    ranking_df = results_df.copy()
    
    # Normalize metrics for ranking (0-1 scale, higher is better)
    key_metrics = ['modularity', 'coverage', 'performance', 'silhouette_score']
    available_metrics = [m for m in key_metrics if m in ranking_df.columns]
    
    if available_metrics:
        for metric in available_metrics:
            min_val = ranking_df[metric].min()
            max_val = ranking_df[metric].max()
            if max_val > min_val:
                ranking_df[f'{metric}_normalized'] = (ranking_df[metric] - min_val) / (max_val - min_val)
            else:
                ranking_df[f'{metric}_normalized'] = 0.5  # All equal
        
        # Invert conductance (lower is better)
        if 'conductance' in ranking_df.columns:
            min_val = ranking_df['conductance'].min()
            max_val = ranking_df['conductance'].max()
            if max_val > min_val:
                ranking_df['conductance_normalized'] = 1 - (ranking_df['conductance'] - min_val) / (max_val - min_val)
            else:
                ranking_df['conductance_normalized'] = 0.5
            available_metrics.append('conductance')
        
        # Calculate composite score
        normalized_cols = [f'{m}_normalized' for m in available_metrics]
        ranking_df['composite_score'] = ranking_df[normalized_cols].mean(axis=1)
        ranking_df = ranking_df.sort_values('composite_score', ascending=False)
    
    # Create and display ranking summary
    ranking_summary = {}
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'Rank':<5} {'Method':<15} {'Composite':<10} {'Communities':<12} {'Modularity':<10} {'Coverage':<10}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(ranking_df.iterrows()):
        rank = i + 1
        method = row['method']
        composite = row.get('composite_score', 0)
        communities = row.get('num_communities', 0)
        modularity = row.get('modularity', 0)
        coverage = row.get('coverage', 0)
        
        ranking_summary[method] = rank
        
        print(f"{rank:<5} {method:<15} {composite:.3f}{'':>4} {communities:<12} "
              f"{modularity:.3f}{'':>4} {coverage:.3f}")
    
    return ranking_summary


def create_ranking_summary(results_df: pd.DataFrame, numeric_cols: List[str], 
                          maximize_metrics: List[str], minimize_metrics: List[str]) -> pd.DataFrame:
    """Create a ranking summary table."""
    print("\n" + "="*80)
    print("RANKING SUMMARY (1 = Best)")
    print("="*80)
    
    ranking_df = results_df.copy()
    
    # Rank each metric (1 = best)
    for metric in numeric_cols:
        if metric in maximize_metrics:
            ranking_df[f'{metric}_rank'] = ranking_df[metric].rank(ascending=False, method='min')
        elif metric in minimize_metrics:
            ranking_df[f'{metric}_rank'] = ranking_df[metric].rank(ascending=True, method='min')
    
    # Calculate average rank
    rank_cols = [col for col in ranking_df.columns if col.endswith('_rank')]
    ranking_df['avg_rank'] = ranking_df[rank_cols].mean(axis=1)
    
    # Sort by average rank and display
    summary = ranking_df[['method', 'avg_rank'] + rank_cols].sort_values('avg_rank')
    print(summary.round(2))
    
    return summary


def create_comprehensive_cross_dataset_plots(alice_results_df, jekyll_results_df, alice_distance_df, jekyll_distance_df):
    """Create comprehensive plotting visualization for cross-dataset comparison."""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    all_metrics_plot = ['modularity', 'silhouette_score', 'coverage', 'performance', 'conductance', 'normalized_cut', 'IntraPW', 'IntraCent', 'Across']
    
    for i, metric in enumerate(all_metrics_plot):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        if metric in ['IntraPW', 'IntraCent', 'Across']:
            alice_data = alice_distance_df.set_index('method')[metric]
            jekyll_data = jekyll_distance_df.set_index('method')[metric]
        else:
            alice_data = alice_results_df.set_index('method')[metric]
            jekyll_data = jekyll_results_df.set_index('method')[metric]
        
        x = range(len(alice_data))
        width = 0.35
        
        bars1 = ax.bar([i - width/2 for i in x], alice_data.values, width, label='Alice', alpha=0.8, color='skyblue')
        bars2 = ax.bar([i + width/2 for i in x], jekyll_data.values, width, label='Jekyll', alpha=0.8, color='lightcoral')
        
        # Add value labels for distance metrics
        if metric in ['IntraPW', 'IntraCent', 'Across']:
            for j, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                height1 = bar1.get_height()
                height2 = bar2.get_height()
                ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                       f'{height1:.3f}', ha='center', va='bottom', fontsize=8)
                ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                       f'{height2:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(alice_data.index, rotation=45, ha='right', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Community Detection Performance: Alice vs Jekyll', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
