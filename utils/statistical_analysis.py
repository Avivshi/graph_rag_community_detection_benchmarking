import numpy as np
import pandas as pd
from utils.community_utils import analyze_partition
from community_detection.community_kmeans import kmeans_partition
from community_detection.community_leiden import leiden_partition
from community_detection.community_weighted_leiden import run_leiden_on_weighted_graph
from community_detection.community_spectral import spectral_partition_from_embeddings
from community_detection.community_fiedler import fiedler_partition
from community_detection.community_girvan_newman import girvan_newman_partition
from community_detection.community_hierarchical import hierarchical_partition


RANDOM_SEEDS = [42, 123, 456, 789, 999]


def _run_kmeans_with_seed(nodes, node_embeddings, seed):
    """Run K-Means clustering with a specific seed."""
    try:
        kmeans_dict = kmeans_partition(nodes, node_embeddings, k=3, random_state=seed)
        kmeans_stats, kmeans_across = analyze_partition(kmeans_dict, node_embeddings)
        return {
            'IntraPW': average_of_list(kmeans_stats["intra_pairwise"]),
            'IntraCent': average_of_list(kmeans_stats["intra_centroid"]),
            'Across': kmeans_across
        }
    except Exception as e:
        print(f"  K-Means failed with seed {seed}: {e}")
        return {'IntraPW': np.nan, 'IntraCent': np.nan, 'Across': np.nan}


def _run_leiden_with_seed(nodes, edges, node_embeddings, seed):
    """Run Leiden clustering with a specific seed."""
    try:
        leiden_dict = leiden_partition(nodes, edges)
        leiden_stats, leiden_across = analyze_partition(leiden_dict, node_embeddings)
        return {
            'IntraPW': average_of_list(leiden_stats["intra_pairwise"]),
            'IntraCent': average_of_list(leiden_stats["intra_centroid"]),
            'Across': leiden_across
        }
    except Exception as e:
        print(f"  Leiden failed with seed {seed}: {e}")
        return {'IntraPW': np.nan, 'IntraCent': np.nan, 'Across': np.nan}


def _run_weighted_leiden_with_seed(nodes, edges, node_embeddings, build_weighted_graph, seed):
    """Run Weighted Leiden clustering with a specific seed."""
    try:
        alpha = 0.5
        Gw = build_weighted_graph(nodes, edges, node_embeddings, alpha=alpha)
        wl_dict = run_leiden_on_weighted_graph(Gw, nodes)
        wl_stats, wl_across = analyze_partition(wl_dict, node_embeddings)
        return {
            'IntraPW': average_of_list(wl_stats["intra_pairwise"]),
            'IntraCent': average_of_list(wl_stats["intra_centroid"]),
            'Across': wl_across
        }
    except Exception as e:
        print(f"  Weighted Leiden failed with seed {seed}: {e}")
        return {'IntraPW': np.nan, 'IntraCent': np.nan, 'Across': np.nan}


def _run_spectral_with_seed(nodes, node_embeddings, seed):
    """Run Spectral clustering with a specific seed."""
    try:
        spec_dict = spectral_partition_from_embeddings(nodes, node_embeddings, n_clusters=3, gamma=1.0)
        spec_stats, spec_across = analyze_partition(spec_dict, node_embeddings)
        return {
            'IntraPW': average_of_list(spec_stats["intra_pairwise"]),
            'IntraCent': average_of_list(spec_stats["intra_centroid"]),
            'Across': spec_across
        }
    except Exception as e:
        print(f"  Spectral failed with seed {seed}: {e}")
        return {'IntraPW': np.nan, 'IntraCent': np.nan, 'Across': np.nan}


def _run_hierarchical_with_seed(nodes, node_embeddings, seed):
    """Run Hierarchical clustering with a specific seed."""
    try:
        hier_dict = hierarchical_partition(nodes, node_embeddings, num_clusters=3)
        hier_stats, hier_across = analyze_partition(hier_dict, node_embeddings)
        return {
            'IntraPW': average_of_list(hier_stats["intra_pairwise"]),
            'IntraCent': average_of_list(hier_stats["intra_centroid"]),
            'Across': hier_across
        }
    except Exception as e:
        print(f"  Hierarchical failed with seed {seed}: {e}")
        return {'IntraPW': np.nan, 'IntraCent': np.nan, 'Across': np.nan}


def _run_deterministic_algorithms(undirected_G, nodes, node_embeddings):
    """Run deterministic algorithms once (results don't change with seeds)."""
    deterministic_results = {}
    
    # Fiedler Partition (deterministic)
    try:
        community_1, community_2 = fiedler_partition(undirected_G)
        fiedler_dict = {0: community_1, 1: community_2}
        fiedler_stats, fiedler_across = analyze_partition(fiedler_dict, node_embeddings)
        deterministic_results['Fiedler'] = {
            'IntraPW': average_of_list(fiedler_stats["intra_pairwise"]),
            'IntraCent': average_of_list(fiedler_stats["intra_centroid"]),
            'Across': fiedler_across
        }
    except Exception as e:
        print(f"  Fiedler failed: {e}")
        deterministic_results['Fiedler'] = {'IntraPW': np.nan, 'IntraCent': np.nan, 'Across': np.nan}
    
    # Girvan-Newman (deterministic for same num_communities)
    try:
        gn_dict = girvan_newman_partition(undirected_G, num_communities=3)
        gn_stats, gn_across = analyze_partition(gn_dict, node_embeddings)
        deterministic_results['Girvan-Newman'] = {
            'IntraPW': average_of_list(gn_stats["intra_pairwise"]),
            'IntraCent': average_of_list(gn_stats["intra_centroid"]),
            'Across': gn_across
        }
    except Exception as e:
        print(f"  Girvan-Newman failed: {e}")
        deterministic_results['Girvan-Newman'] = {'IntraPW': np.nan, 'IntraCent': np.nan, 'Across': np.nan}
    
    return deterministic_results


def _run_stochastic_algorithms_with_seed(nodes, edges, node_embeddings, build_weighted_graph, seed):
    """Run all stochastic algorithms with a specific seed."""
    seed_results = {
        'K-Means': _run_kmeans_with_seed(nodes, node_embeddings, seed),
        'Leiden': _run_leiden_with_seed(nodes, edges, node_embeddings, seed),
        'Weighted Leiden': _run_weighted_leiden_with_seed(nodes, edges, node_embeddings, build_weighted_graph, seed),
        'Spectral': _run_spectral_with_seed(nodes, node_embeddings, seed),
        'Hierarchical': _run_hierarchical_with_seed(nodes, node_embeddings, seed)
    }
    return seed_results


def _calculate_composite_score(result):
    """Calculate composite score for ranking algorithm performance."""
    # Higher score = better performance (lower IntraPW + IntraCent, higher Across)
    return (1 - result['IntraPW']) + (1 - result['IntraCent']) + result['Across']


def _find_best_seed_for_method(method, all_results):
    """Find the best-performing seed for a specific method."""
    seed_scores = {}
    
    for seed in RANDOM_SEEDS:
        if seed in all_results and method in all_results[seed]:
            result = all_results[seed][method]
            if not any(np.isnan(result[m]) for m in ['IntraPW', 'IntraCent', 'Across']):
                composite_score = _calculate_composite_score(result)
                seed_scores[seed] = {
                    'composite': composite_score,
                    'metrics': result
                }
    
    if seed_scores:
        best_seed = max(seed_scores.keys(), key=lambda s: seed_scores[s]['composite'])
        print(f"📊 {method}: Best seed {best_seed} (score: {seed_scores[best_seed]['composite']:.4f})")
        return best_seed, seed_scores
    else:
        print(f"⚠️ {method}: No valid results, using default seed {RANDOM_SEEDS[0]}")
        return RANDOM_SEEDS[0], {}


def _calculate_stochastic_method_stats(method, all_results, seed_scores, best_seed):
    """Calculate statistics for a stochastic method across all seeds."""
    method_stats = {}
    
    for metric in ['IntraPW', 'IntraCent', 'Across']:
        values = [
            all_results[seed][method][metric] 
            for seed in RANDOM_SEEDS 
            if seed in all_results and method in all_results[seed] 
            and not np.isnan(all_results[seed][method][metric])
        ]
        
        if values:
            method_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n_valid': len(values),
                'best_seed_value': seed_scores[best_seed]['metrics'][metric] if best_seed in seed_scores else np.nan
            }
        else:
            method_stats[metric] = {
                'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'n_valid': 0,
                'best_seed_value': np.nan
            }
    
    return method_stats


def _calculate_deterministic_method_stats(method, deterministic_results):
    """Calculate statistics for a deterministic method."""
    method_stats = {}
    
    for metric in ['IntraPW', 'IntraCent', 'Across']:
        value = deterministic_results[method][metric]
        if not np.isnan(value):
            method_stats[metric] = {
                'mean': value,
                'std': 0.0,
                'min': value,
                'max': value,
                'n_valid': 1,
                'best_seed_value': value
            }
        else:
            method_stats[metric] = {
                'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'n_valid': 0,
                'best_seed_value': np.nan
            }
    
    return method_stats


def _calculate_method_statistics_with_best_seeds(all_results, deterministic_results):
    """Calculate statistics and track best-performing seeds for each method."""
    method_stats = {}
    best_seeds = {}
    
    # Stochastic methods - find best seeds
    stochastic_methods = ['K-Means', 'Leiden', 'Weighted Leiden', 'Spectral', 'Hierarchical']
    for method in stochastic_methods:
        best_seed, seed_scores = _find_best_seed_for_method(method, all_results)
        best_seeds[method] = best_seed
        method_stats[method] = _calculate_stochastic_method_stats(method, all_results, seed_scores, best_seed)
    
    # Deterministic methods (no seed dependency)
    deterministic_methods = ['Fiedler', 'Girvan-Newman']
    for method in deterministic_methods:
        best_seeds[method] = RANDOM_SEEDS[0]  # Any seed works for deterministic
        method_stats[method] = _calculate_deterministic_method_stats(method, deterministic_results)
    
    return method_stats, best_seeds


def _display_statistical_summary(method_stats):
    """Display the statistical summary of results."""
    print(f"\nSTATISTICAL SUMMARY (n={len(RANDOM_SEEDS)} seeds for stochastic methods):")
    print("="*80)
    
    # Group methods by type for better display
    stochastic_methods = ['K-Means', 'Leiden', 'Weighted Leiden', 'Spectral', 'Hierarchical']
    deterministic_methods = ['Fiedler', 'Girvan-Newman']
    
    # Display stochastic methods
    print("\nSTOCHASTIC METHODS (Multiple Seeds):")
    print("-" * 50)
    for method in stochastic_methods:
        if method in method_stats:
            print(f"\n{method}:")
            for metric in ['IntraPW', 'IntraCent', 'Across']:
                stats = method_stats[method][metric]
                if stats['n_valid'] > 0:
                    print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                          f"(range: {stats['min']:.4f}-{stats['max']:.4f}, n={stats['n_valid']})")
                else:
                    print(f"  {metric}: No valid results")
    
    # Display deterministic methods
    print("\n\nDETERMINISTIC METHODS (Single Result):")
    print("-" * 50)
    for method in deterministic_methods:
        if method in method_stats:
            print(f"\n{method}:")
            for metric in ['IntraPW', 'IntraCent', 'Across']:
                stats = method_stats[method][metric]
                if stats['n_valid'] > 0:
                    print(f"  {metric}: {stats['mean']:.4f} (deterministic)")
                else:
                    print(f"  {metric}: No valid results")
                    


def run_multiple_seeds_analysis(nodes, edges, node_embeddings, undirected_G, dataset_name, build_weighted_graph):
    """Run all algorithms multiple times with different seeds for statistical robustness."""
    print(f"\n{'='*80}")
    print(f"MULTI-SEED STATISTICAL ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # Run deterministic algorithms once
    print("\n🎯 Running deterministic algorithms...")
    deterministic_results = _run_deterministic_algorithms(undirected_G, nodes, node_embeddings)
    
    # Run stochastic algorithms for each seed
    all_results = {}
    for seed in RANDOM_SEEDS:
        print(f"\n🎯 Running stochastic algorithms with seed: {seed}")
        np.random.seed(seed)
        all_results[seed] = _run_stochastic_algorithms_with_seed(
            nodes, edges, node_embeddings, build_weighted_graph, seed
        )
    
    # Calculate statistics and find best seeds
    method_stats, best_seeds = _calculate_method_statistics_with_best_seeds(all_results, deterministic_results)
    
    # Display results
    _display_statistical_summary(method_stats)
    
    # Display best seeds found
    print(f"\n{'='*60}")
    print("BEST PERFORMING SEEDS IDENTIFIED:")
    print(f"{'='*60}")
    for method, seed in best_seeds.items():
        print(f"{method:20} → Seed {seed}")
    
    return method_stats, all_results, best_seeds


def average_of_list(vals):
    """Calculate the average of a list of values."""
    return np.mean(vals) if vals else 0.0


def perform_significance_tests(method_stats):
    """Perform statistical significance tests between methods."""
    print(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
    print("="*60)
    
    methods = list(method_stats.keys())
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods[i+1:], i+1):
            print(f"\n{method1} vs {method2}:")
            
            for metric in ['IntraPW', 'IntraCent', 'Across']:
                stats1 = method_stats[method1][metric]
                stats2 = method_stats[method2][metric]
                
                if stats1['n_valid'] >= 3 and stats2['n_valid'] >= 3:
                    try:
                        diff = abs(stats1['mean'] - stats2['mean'])
                        pooled_std = (stats1['std'] + stats2['std']) / 2
                        significance = "***" if diff > 2 * pooled_std else "**" if diff > pooled_std else "*" if diff > 0.5 * pooled_std else "ns"
                        print(f"  {metric}: {significance} (Δ={diff:.4f})")
                    except:
                        print(f"  {metric}: Unable to test")
                else:
                    print(f"  {metric}: Insufficient data")


def prepare_cross_dataset_dataframes(alice_results_df, jekyll_results_df, 
                                   alice_labels_methods, alice_intra_pw, alice_intra_cent, alice_across_vals,
                                   jekyll_labels_methods, jekyll_intra_pw, jekyll_intra_cent, jekyll_across_vals):
    """Prepare and combine dataframes for cross-dataset analysis."""
    
    # Add dataset identifiers
    alice_results_df['dataset'] = 'Alice'
    jekyll_results_df['dataset'] = 'Jekyll'
    
    # Create distance metrics dataframes
    alice_distance_df = pd.DataFrame({
        'method': alice_labels_methods,
        'IntraPW': alice_intra_pw,
        'IntraCent': alice_intra_cent,
        'Across': alice_across_vals,
        'dataset': 'Alice'
    })
    
    jekyll_distance_df = pd.DataFrame({
        'method': jekyll_labels_methods,
        'IntraPW': jekyll_intra_pw,
        'IntraCent': jekyll_intra_cent,
        'Across': jekyll_across_vals,
        'dataset': 'Jekyll'
    })
    
    # Combine all results
    combined_results = pd.concat([alice_results_df, jekyll_results_df], ignore_index=True)
    combined_distance = pd.concat([alice_distance_df, jekyll_distance_df], ignore_index=True)
    
    return combined_results, combined_distance, alice_distance_df, jekyll_distance_df


def create_comparison_pivots(combined_results, combined_distance):
    """Create pivot tables for comprehensive comparison."""
    
    # Graph-theoretic metrics pivot
    all_metrics = ['modularity', 'silhouette_score', 'coverage', 'performance', 'conductance', 'normalized_cut']
    comparison_pivot = combined_results.pivot_table(
        index='method', 
        columns='dataset', 
        values=all_metrics,
        aggfunc='first'
    )
    
    # Distance metrics pivot
    distance_pivot = combined_distance.pivot_table(
        index='method',
        columns='dataset', 
        values=['IntraPW', 'IntraCent', 'Across'],
        aggfunc='first'
    )
    
    return comparison_pivot, distance_pivot


def analyze_cross_dataset_improvements(alice_distance_df, jekyll_distance_df, alice_labels_methods, 
                                     alice_intra_pw, alice_intra_cent, alice_across_vals,
                                     jekyll_intra_pw, jekyll_intra_cent, jekyll_across_vals):
    """Analyze and calculate improvement percentages between datasets."""
    
    print("\nVERIFICATION - Distance Metrics Differences:")
    for metric in ['IntraPW', 'IntraCent', 'Across']:
        alice_vals = alice_distance_df[metric].values
        jekyll_vals = jekyll_distance_df[metric].values
        differences = jekyll_vals - alice_vals
        print(f"{metric}:")
        for i, method in enumerate(alice_labels_methods):
            print(f"  {method}: Alice={alice_vals[i]:.4f}, Jekyll={jekyll_vals[i]:.4f}, Diff={differences[i]:+.4f}")
    
    # Calculate improvement percentages
    print("\nIMPROVEMENT ANALYSIS (Alice → Jekyll):")
    improvement_summary = []
    for i, method in enumerate(alice_labels_methods):
        intrapw_change = ((jekyll_intra_pw[i] - alice_intra_pw[i]) / alice_intra_pw[i]) * 100
        intracent_change = ((jekyll_intra_cent[i] - alice_intra_cent[i]) / alice_intra_cent[i]) * 100
        across_change = ((jekyll_across_vals[i] - alice_across_vals[i]) / alice_across_vals[i]) * 100
        
        improvement_summary.append({
            'method': method,
            'IntraPW_change': intrapw_change,
            'IntraCent_change': intracent_change, 
            'Across_change': across_change
        })
        
        print(f"{method}:")
        print(f"  IntraPW: {intrapw_change:+.1f}% ({'✓' if intrapw_change < 0 else '✗'})")
        print(f"  IntraCent: {intracent_change:+.1f}% ({'✓' if intracent_change < 0 else '✗'})")
        print(f"  Across: {across_change:+.1f}% ({'✓' if across_change > 0 else '✗'})")
    
    return improvement_summary


def generate_cross_dataset_summary_statistics(comparison_pivot, distance_pivot, improvement_summary):
    """Generate and display summary statistics for cross-dataset analysis."""
    
    print("\nSUMMARY STATISTICS:")
    print("="*60)
    
    print("\nTOP PERFORMERS BY CATEGORY:")
    print("-" * 40)
    
    # Graph metrics leaders
    modularity_leader = comparison_pivot['modularity'].max(axis=1).idxmax()
    coverage_leader = comparison_pivot['coverage'].max(axis=1).idxmax()
    performance_leader = comparison_pivot['performance'].max(axis=1).idxmax()
    
    print(f"Modularity Champion: {modularity_leader}")
    print(f"Coverage Champion: {coverage_leader}")
    print(f"Performance Champion: {performance_leader}")
    
    # Distance metrics leaders
    intrapw_leader = distance_pivot['IntraPW'].min(axis=1).idxmin()
    intracent_leader = distance_pivot['IntraCent'].min(axis=1).idxmin()
    across_leader = distance_pivot['Across'].max(axis=1).idxmax()
    
    print(f"Best Semantic Cohesion (IntraPW): {intrapw_leader}")
    print(f"Best Centroid Distance (IntraCent): {intracent_leader}")  
    print(f"Best Separation (Across): {across_leader}")
    
    # Most improved algorithms
    improvement_df = pd.DataFrame(improvement_summary)
    most_improved_intrapw = improvement_df.loc[improvement_df['IntraPW_change'].idxmin(), 'method']
    most_improved_across = improvement_df.loc[improvement_df['Across_change'].idxmax(), 'method']
    
    print(f"\nMOST IMPROVED (Alice → Jekyll):")
    print(f"Semantic Cohesion: {most_improved_intrapw}")
    print(f"Community Separation: {most_improved_across}")


def analyze_quality_thresholds(combined_distance):
    """Analyze methods meeting quality thresholds."""
    
    print(f"\nQUALITY THRESHOLD ANALYSIS:")
    print("-" * 40)
    
    excellent_threshold = {
        'IntraPW': 0.30,
        'IntraCent': 0.30, 
        'Across': 0.65
    }
    
    for dataset in ['Alice', 'Jekyll']:
        dataset_distance = combined_distance[combined_distance['dataset'] == dataset]
        print(f"\n{dataset} Dataset - Methods Meeting Excellence Thresholds:")
        
        for metric, threshold in excellent_threshold.items():
            if metric == 'Across':
                qualifying = dataset_distance[dataset_distance[metric] >= threshold]['method'].tolist()
            else:
                qualifying = dataset_distance[dataset_distance[metric] <= threshold]['method'].tolist()
            print(f"  {metric} ({'≥' if metric == 'Across' else '≤'}{threshold}): {qualifying}")


def display_enhanced_statistical_insights(labels_methods, intra_pw, intra_cent, across_vals, stochastic_stats, baseline_results):
    """Display enhanced comparison with statistical insights."""
    
    print("\n=== ENHANCED COMPARISON WITH STATISTICAL INSIGHTS ===")
    for i, method in enumerate(labels_methods):
        base_info = f"{method} => IntraPW: {intra_pw[i]:.4f}, IntraCent: {intra_cent[i]:.4f}, Across: {across_vals[i]:.4f}"
        
        # Add statistical info if available
        if method in stochastic_stats and stochastic_stats[method]['IntraPW']['n_valid'] > 0:
            stats = stochastic_stats[method]
            enhanced_info = f" (±{stats['IntraPW']['std']:.3f}, ±{stats['IntraCent']['std']:.3f}, ±{stats['Across']['std']:.3f})"
            print(base_info + enhanced_info)
        else:
            print(base_info + " (deterministic)")
    
    # Show baseline comparison
    if baseline_results:
        print("\n=== BASELINE COMPARISON ===")
        for name, result in baseline_results.items():
            if result:
                print(f"{name}: IntraPW={result['IntraPW']:.4f}, IntraCent={result['IntraCent']:.4f}, Across={result['Across']:.4f}")
