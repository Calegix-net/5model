import pandas as pd
import glob
import os

# 結合後のファイル名
output_file = 'dataset.csv'

# 結合後のデータフレームを格納するリスト
dataframes = []

# 既存の最大のRun_IDを取得（初期値は-1）
if os.path.exists(output_file):
    combined_df = pd.read_csv(output_file)
    max_run_id = combined_df['Run_ID'].max()
    dataframes.append(combined_df)
else:
    max_run_id = -1

# 結合するCSVファイルのパターンを指定（適切なファイル名に合わせて変更してください）
csv_files = glob.glob('*.csv')  # 例："data_"で始まるCSVファイル

# 各CSVファイルを処理
for file in csv_files:
    df = pd.read_csv(file)

    # Run_IDを調整
    df['Run_ID'] = df['Run_ID'] + max_run_id + 1

    # データフレームをリストに追加
    dataframes.append(df)

    # 新しい最大のRun_IDを更新
    max_run_id = df['Run_ID'].max()

# 全てのデータフレームを結合
combined_df = pd.concat(dataframes, ignore_index=True)

# 結合後のデータフレームを保存
combined_df.to_csv(output_file, index=False)