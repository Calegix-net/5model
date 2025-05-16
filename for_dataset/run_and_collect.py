import os
import shutil
import subprocess
import pandas as pd

# main.pyの実行回数
NUM_RUNS = 100  # 実行回数を指定

# main.pyで作成される結果ディレクトリとファイルの情報
result_dir = 'result(10c10r)'
weight_dir = 'weight_pth_file_normal(10c10r)'
summary_dir = 'summary_results_normal(10c10r)'
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

# main.pyを実行し、all_layers_summary.csvを収集する
def run_main_and_collect(run_id):
    print(f"--- 実行 {run_id + 1} ---")
    
    # main.pyを実行
    subprocess.run(['python', 'main.py'], check=True)
    
    # 実行が成功したら、summaryファイルを読み込む
    summary_path = os.path.join(summary_dir, summary_file)
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df['Run ID'] = run_id  # 実行回数の識別子を追加
        return df
    else:
        print(f"Error: {summary_path} が存在しません")
        return None

# データセットを追記保存する関数
def append_to_final_dataset(df, header):
    # CSVに追記モードで書き込み（headerは最初の1回のみ書き込む）
    df.to_csv(final_dataset_file, mode='a', index=False, header=header)

# データセットを作成する関数
def create_final_dataset():
    # 最初の1回はヘッダーを含める
    header = True
    
    # 指定回数 main.py を実行し、データを収集
    for run_id in range(NUM_RUNS):
        df = run_main_and_collect(run_id)
        if df is not None:
            # 取得したデータをファイルに追記
            append_to_final_dataset(df, header=header)
            header = False  # 最初の1回のみヘッダーを含める
        
        # ディレクトリとファイルのクリーンアップ
        cleanup()

# 実行
if __name__ == "__main__":
    create_final_dataset()