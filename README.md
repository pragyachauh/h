import numpy as np
from scipy.signal import find_peaks

def improved_peaks_and_valleys(data, column_name, prominence=None, distance=1):
    """
    Enhanced function to find peaks and valleys that handles monotonically increasing/decreasing sequences.
    
    Args:
        data: DataFrame containing the metrics
        column_name: Name of the column to analyze
        prominence: Optional parameter for find_peaks to filter peaks based on prominence
        distance: Minimum distance between peaks
        
    Returns:
        Dictionary with 'peaks' and 'valleys' lists
    """
    y = data[column_name].values
    
    # For default find_peaks behavior
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)
    valleys, _ = find_peaks(-y, prominence=prominence, distance=distance)
    
    # Handle special cases for different metrics
    if column_name == 'silhouette_score':  # Higher is better
        # If no peaks and values are generally increasing, consider the last point as a peak
        if len(peaks) == 0:
            if y[-1] == max(y) or sum(y[i] < y[i+1] for i in range(len(y)-1)) >= len(y) * 0.7:
                peaks = np.array([len(y) - 1])
    
    elif column_name == 'calinski_harabasz_index':  # Higher is better
        # If no peaks and the last value is significantly higher, consider it a peak
        if len(peaks) == 0:
            if y[-1] == max(y) or y[-1] > np.mean(y) * 1.1:
                peaks = np.array([len(y) - 1])
    
    elif column_name == 'davies_bouldin_index':  # Lower is better
        # If no valleys and values are generally decreasing, consider the last point as a valley
        if len(valleys) == 0:
            if y[-1] == min(y) or sum(y[i] > y[i+1] for i in range(len(y)-1)) >= len(y) * 0.7:
                valleys = np.array([len(y) - 1])
    
    print(f"{column_name} Peaks: {peaks}")
    print(f"{column_name} Valleys: {valleys}")
    
    results = {
        "peaks": peaks.tolist(),
        "valleys": valleys.tolist()
    }
    
    return results

def find_optimal_score(df):
    """
    Find optimal score using enhanced peaks and valleys detection with fallback options
    
    Args:
        df: DataFrame with clustering metrics
        
    Returns:
        Tuple of (best_cluster_index, best_score, best_row)
    """
    # Find peaks and valleys with enhanced function
    ss_results = improved_peaks_and_valleys(df, 'silhouette_score')
    ch_results = improved_peaks_and_valleys(df, 'calinski_harabasz_index')
    db_results = improved_peaks_and_valleys(df, 'davies_bouldin_index')
    
    ss_peaks = ss_results['peaks']
    ch_peaks = ch_results['peaks']
    db_valleys = db_results['valleys']
    
    print("Silhouette Score Peaks:", ss_peaks)
    print("Calinski-Harabasz Index Peaks:", ch_peaks)
    print("Davies-Bouldin Index Valleys:", db_valleys)
    
    # First approach: Look for exact alignment of peaks/valleys
    optimal_combination = {}
    for ss_peak in ss_peaks:
        if ss_peak in ch_peaks and ss_peak in db_valleys:
            silhouette = df.iloc[ss_peak]['silhouette_score']
            ch_score = df.iloc[ss_peak]['calinski_harabasz_index']
            db_score = df.iloc[ss_peak]['davies_bouldin_index']
            optimal_score = (silhouette * db_score) / ch_score
            optimal_combination[ss_peak] = optimal_score
    
    # If no exact alignment found, use fallback strategies
    if not optimal_combination:
        print("No exact alignment found. Using fallback approach...")
        
        # Fallback 1: Use the multipliedScore column if exists
        if 'multipliedScore' in df.columns:
            best_idx = df['multipliedScore'].idxmax()
            best_score = df.loc[best_idx, 'multipliedScore']
            best_row = df.iloc[best_idx]
            return best_idx, best_score, best_row
        
        # Fallback 2: Weighted approach based on all peaks/valleys
        all_special_indices = set(ss_peaks + ch_peaks + db_valleys)
        for idx in all_special_indices:
            if idx < len(df):
                silhouette = df.iloc[idx]['silhouette_score']
                ch_score = df.iloc[idx]['calinski_harabasz_index']
                db_score = df.iloc[idx]['davies_bouldin_index']
                
                # Calculate a weighted score
                is_ss_peak = idx in ss_peaks
                is_ch_peak = idx in ch_peaks
                is_db_valley = idx in db_valleys
                
                peak_count = is_ss_peak + is_ch_peak + is_db_valley
                if peak_count > 0:
                    weight = peak_count / 3  # Higher weight for indices that appear in multiple metrics
                    weighted_score = (silhouette * (1/db_score) * ch_score) * weight
                    optimal_combination[idx] = weighted_score
        
        # Fallback 3: If still no good options, use the row with best silhouette score
        if not optimal_combination:
            best_ss_idx = df['silhouette_score'].idxmax()
            silhouette = df.loc[best_ss_idx, 'silhouette_score']
            ch_score = df.loc[best_ss_idx, 'calinski_harabasz_index']
            db_score = df.loc[best_ss_idx, 'davies_bouldin_index']
            optimal_score = (silhouette * db_score) / ch_score
            optimal_combination[best_ss_idx] = optimal_score
    
    best_cluster = max(optimal_combination, key=optimal_combination.get)
    best_score = optimal_combination[best_cluster]
    best_row = df.iloc[best_cluster]
    
    print("Optimal Row:")
    print(best_row)
    
    return best_cluster, best_score, best_row

# Example of how to use this in your main function:
"""
def main():
    csv_path = 'Members_Small.csv'
    combinations = [(2, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40)]
    df = append_job_ids(csv_path, combinations)
    best_cluster, best_score, best_row = find_optimal_score(df)
    print(f"Best cluster: {best_cluster}, Best score: {best_score}")
    print(f"Best row: {best_row}")

if __name__ == '__main__':
    main()
"""
