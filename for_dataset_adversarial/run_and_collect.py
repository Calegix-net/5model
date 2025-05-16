import os
import shutil
import subprocess
import pandas as pd
import time

# main_adversarial.pyの実行回数
NUM_RUNS = 100  # 実行回数を指定
MAX_RETRIES = 5  # リトライ回数の上限

# main_random.pyで作成される結果ディレクトリとファイルの情報
result_dir = 'result(10c10r)'
weight_dir = 'weight_pth_file_adversarial(10c10r)'
summary_dir = 'summary_results_adversarial(10c10r)'
summary_file = 'all_layers_summary.csv'

# データセットを保存するファイル名
def get_dataset_filename():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f'final_dataset_10%_{timestamp}.csv'

# ディレクトリやファイルの削除
def cleanup():
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    if os.path.exists(weight_dir):
        shutil.rmtree(weight_dir)
    if os.path.exists(summary_dir):
        shutil.rmtree(summary_dir)

# main_adversarial.pyを実行し、all_layers_summary.csvを収集する
def run_main_and_collect(run_id):
    print(f"--- 実行 {run_id + 1} ---")
    
    for attempt in range(MAX_RETRIES):
        # 実行前にクリーンアップを行う
        cleanup()
        
        try:
            # main_adversarial.pyを実行
            subprocess.run(['python', 'main_adversarial.py'], check=True)
            
            # 実行が成功したら、summaryファイルを読み込む
            summary_path = os.path.join(summary_dir, summary_file)
            if os.path.exists(summary_path):
                df = pd.read_csv(summary_path)
                df['Run ID'] = run_id  # 実行回数の識別子を追加
                return df
            else:
                print(f"Error: {summary_path} が存在しません")
                return None
            
        except subprocess.CalledProcessError:
            print(f"Error: main_adversarial.py の実行に失敗しました (試行 {attempt + 1}/{MAX_RETRIES})")
            time.sleep(10)  # 少し待機してから再試行
            
    print(f"Error: main_adversarial.py の実行に {MAX_RETRIES} 回失敗しました。次の実行に進みます。")
    return None

# データセットを作成する関数
def create_final_dataset():
    header = True  # 最初の1回だけヘッダーを含める
    dataset_file = get_dataset_filename()  # タイムスタンプ付きのファイル名を取得
    
    # 指定回数 main_adversarial.py を実行し、データを収集
    for run_id in range(NUM_RUNS):
        df = run_main_and_collect(run_id)
        if df is not None:
            # 取得したデータをすぐにCSVに追記
            df.to_csv(dataset_file, mode='a', index=False, header=header)
            header = False  # 最初の1回だけヘッダーを含める
        
        # ディレクトリとファイルのクリーンアップ
        cleanup()

# 実行
if __name__ == "__main__":
    create_final_dataset()