import pandas as pd
import requests
import time
import numpy as np
from scipy.signal import find_peaks

def send_post(csv_path, minClusterSize, maxClusterSize):
    url = 'https://klusteringfastapi.t3ode-lnx-019.tzhealthcare.com/cluster-data/'

    with open(csv_path, 'rb') as csv_file:
        files = [('file', ('file', csv_file, 'application/octet-stream'))]
        headers = {'accept': 'application/json'}
        params = {
            'min_clusters': minClusterSize,
            'max_clusters': maxClusterSize,
            'selected_algorithm': 'K-means with Silhouette'
        }
        response = requests.post(url, headers=headers, data=params, files=files)
        response_json = response.json()
        job_id = response_json.get("job_id", None)

    return job_id

def get_results(job_id):
    url = f"https://klusteringfastapi.t3ode-lnx-019.tzhealthcare.com/get-results-json/{job_id}"
    headers = {'accept': 'application/json'}
    response = requests.get(url, headers=headers)
    response_json = response.json()

    silhouette_score = response_json.get("silhouette_score", None)
    calinski_harabasz_index = response_json.get("calinski_harabasz_score", None)
    davies_bouldin_index = response_json.get("davies_bouldin_score", None)
    n_clusters = response_json.get("n_clusters", None)

    return silhouette_score, calinski_harabasz_index, davies_bouldin_index, n_clusters

def peaks_and_valleys(data, column_name):
    y = data[column_name].values
    x = np.arange(len(y))

    peaks, _ = find_peaks(y)  # Detect peaks without prominence
    valleys, _ = find_peaks(-y)  # Detect valleys without prominence

    print(f"{column_name} Peaks: {peaks}")
    print(f"{column_name} Valleys: {valleys}")

    results = {
        "peaks": peaks.tolist(),
        "valleys": valleys.tolist()
    }

    return results

def to_dataframe_csv(scores, job_ids, combinations):
    df = pd.DataFrame(scores, columns=['silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index', 'n_clusters'])
    df['minClusterSize'] = [comb[0] for comb in combinations]
    df['maxClusterSize'] = [comb[1] for comb in combinations]
    df['job_id'] = job_ids
    df['multipliedScore'] = df['silhouette_score'] * df['davies_bouldin_index'] / df['calinski_harabasz_index']
    df.to_csv('output.csv', index=False)
    print("Scores DataFrame:")
    print(df)
    return df

def append_job_ids(csv_path, combinations):
    job_ids = []
    scores = []
    for combination in combinations:
        minClusterSize, maxClusterSize = combination
        job_id = send_post(csv_path, minClusterSize, maxClusterSize)
        job_ids.append(job_id)

    sleep_time = 10 * len(combinations)
    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)

    for job_id in job_ids:
        scores.append(get_results(job_id))

    return to_dataframe_csv(scores, job_ids, combinations)

def find_optimal_score(df):
    ss_results = peaks_and_valleys(df, 'silhouette_score')
    ch_results = peaks_and_valleys(df, 'calinski_harabasz_index')
    db_results = peaks_and_valleys(df, 'davies_bouldin_index')

    ss_peaks = ss_results['peaks']
    ch_peaks = ch_results['peaks']
    db_valleys = db_results['valleys']

    print("Silhouette Score Peaks:", ss_peaks)
    print("Calinski-Harabasz Index Peaks:", ch_peaks)
    print("Davies-Bouldin Index Valleys:", db_valleys)

    # Checking for peaks and valleys alignment
    if not ss_peaks or not ch_peaks or not db_valleys:
        print("No optimal combination found. Peaks and valleys might not align.")
        raise ValueError("No optimal combination found. Check the peaks and valleys.")

    optimal_combination = {}
    for ss_peak in ss_peaks:
        if ss_peak in ch_peaks and ss_peak in db_valleys:
            silhouette = df.iloc[ss_peak]['silhouette_score']
            ch_score = df.iloc[ss_peak]['calinski_harabasz_index']
            db_score = df.iloc[ss_peak]['davies_bouldin_index']
            optimal_score = (silhouette * db_score) / ch_score
            optimal_combination[ss_peak] = optimal_score

    if not optimal_combination:
        print("No optimal combination found. Peaks and valleys might not align.")
        raise ValueError("No optimal combination found. Check the peaks and valleys.")

    best_cluster = max(optimal_combination, key=optimal_combination.get)
    best_score = optimal_combination[best_cluster]
    best_row = df.iloc[best_cluster]

    print("Optimal Row:")
    print(best_row)

    return best_cluster, best_score, best_row

def main():
    csv_path = 'Members_Small.csv'
    combinations = [(2, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40)]
    df = append_job_ids(csv_path, combinations)
    best_cluster, best_score, best_row = find_optimal_score(df)
    print(f"Best cluster: {best_cluster}, Best score: {best_score}")
    print(f"Best row: {best_row}")

if __name__ == '__main__':
    main()


    Scores DataFrame:
   silhouette_score  calinski_harabasz_index  ...                                job_id  multipliedScore
0          0.654713               760.294458  ...  f5f75db4-a202-4b29-9f7e-c2ae068aff0c         0.000862
1          0.710910               868.747646  ...  dd8509eb-98c8-4d36-81c3-4b1b56c2a623         0.000745
2          0.723418               785.031879  ...  6d7ff5d5-fa06-4627-bf03-6ae9ec77410c         0.000836
3          0.739936               763.016000  ...  6d674ecd-3fe3-4b14-944e-de2a4f5c0c80         0.000805
4          0.758273               770.989089  ...  4a44b403-1a21-45b1-a496-adc2344a6ff4         0.000861
5          0.781344               826.514275  ...  25d6cb1e-3182-431a-9860-a0023173d0c0         0.000686
6          0.804376               880.556179  ...  3342f234-f461-4ed9-9355-61d41095448f         0.000596
7          0.823501               980.066826  ...  6ca28cfb-3214-4db9-8432-1ad0ce9f7eac         0.000543

[8 rows x 8 columns]
silhouette_score Peaks: []
silhouette_score Valleys: []
calinski_harabasz_index Peaks: [1]
calinski_harabasz_index Valleys: [3]
davies_bouldin_index Peaks: [4]
davies_bouldin_index Valleys: [3]
Silhouette Score Peaks: []
Calinski-Harabasz Index Peaks: [1]
Davies-Bouldin Index Valleys: [3]
No optimal combination found. Peaks and valleys might not align.
Traceback (most recent call last):
  File "C:\Users\967302\repos\OSPAI_SCD_VERIFY\test.py", line 131, in <module>
    main()
  File "C:\Users\967302\repos\OSPAI_SCD_VERIFY\test.py", line 126, in main
    best_cluster, best_score, best_row = find_optimal_score(df)
                                         ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\967302\repos\OSPAI_SCD_VERIFY\test.py", line 98, in find_optimal_score
    raise ValueError("No optimal combination found. Check the peaks and valleys.")
ValueError: No optimal combination found. Check the peaks and valleys.
PS C:\Users\967302\repos\OSPAI_SCD_VERIFY>
