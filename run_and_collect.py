import os
import shutil
import subprocess
import pandas as pd

# main_new.pyの実行回数
NUM_RUNS = 2  # 実行回数を指定

# main_new.pyで作成される結果ディレクトリとファイルの情報
result_dir = 'result(2c2r)'
weight_dir = 'weight_pth_file_normal(2c2r)'
summary_dir = 'summary_results_normal(2c2r)'
summary_file = 'all_layers_summary.csv'

# データセットを保存するファイル名
final_dataset_file = 'final_dataset.csv'

# ディレクトリやファイルの削除
def cleanup():
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    if os.path.exists(weight_dir):
        shutil.rmtree(weight_dir)
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)

# main_new.pyを実行し、all_layers_summary.csvを収集する
def run_main_new_and_collect(run_id):
    print(f"--- 実行 {run_id + 1} ---")
    
    # main_new.pyを実行
    subprocess.run(['python', 'main_new.py'], check=True)
    
    # 実行が成功したら、summaryファイルを読み込む
    summary_path = os.path.join(summary_dir, summary_file)
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df['Run ID'] = run_id  # 実行回数の識別子を追加
        return df
    else:
        print(f"Error: {summary_path} が存在しません")
        return None

# データセットを作成する関数
def create_final_dataset():
    all_data = []  # すべてのデータをここに格納する
    
    # 指定回数 main_new.py を実行し、データを収集
    for run_id in range(NUM_RUNS):
        df = run_main_new_and_collect(run_id)
        if df is not None:
            all_data.append(df)
        
        # ディレクトリとファイルのクリーンアップ
        cleanup()

    # すべてのデータを1つにまとめる
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(final_dataset_file, index=False)
        print(f"データセットを {final_dataset_file} に保存しました")
    else:
        print("データが収集されませんでした")

# 実行
if __name__ == "__main__":
    create_final_dataset()