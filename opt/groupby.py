import pandas as pd
import argparse
import numpy as np

def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_cluster', type=int, default=5)
    args = parser.parse_args()
    return args

def main():
    num_clusters = cli_args().num_cluster
    filename = f"/workspaces/doc_analysis/output/result-with-{num_clusters}.csv"
    df = pd.read_csv(filename, header=None, names=['cluster', 'category'])
    df['exist'] = [1 for i in range(len(df))]
    df2 = df.groupby(['cluster', 'category']).count().sort_values(by=['cluster','exist'],ascending=False)
    df2.to_csv(f"/workspaces/doc_analysis/output/groupby-{num_clusters}.csv")

if __name__ == "__main__":
    main()