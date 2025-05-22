import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

result_directory = 'result_malicious(10c10r)'
summary_files = glob.glob(os.path.join(result_directory, "*_summary.csv"))

# グラフの初期設定
plt.figure(figsize=(10, 6))

# 各 summary ファイルを読み込み、点プロット
for summary_file in summary_files:
    df = pd.read_csv(summary_file)
    layer_name = os.path.basename(summary_file).replace('_summary.csv', '')
    plt.scatter(df['Round'], df['Mean Variance'], label=layer_name)

# グラフの設定
plt.xlabel('Round')
plt.ylabel('Mean Variance')
plt.title('Mean Variance by Round for Different Layers')
plt.legend(loc='best')
plt.grid(True)

# グラフを保存
graph_file = os.path.join(result_directory, "mean_variance_by_round.png")
plt.savefig(graph_file)
plt.show()