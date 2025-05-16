import torch
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer

# モデルとトークナイザーのロード
model_name = "distilbert-base-uncased"
model = AutoModel.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 入力シーケンス
sequence = "This novel is very popular."
inputs = tokenizer(sequence, return_tensors="pt")

# モデルの出力（注意マップを含む）
outputs = model(**inputs)
attentions = outputs.attentions

# 層とヘッドの数を取得
num_layers = len(model.transformer.layer)
num_heads = model.config.num_attention_heads

print(f"Number of layers: {num_layers}")
print(f"Number of heads per layer: {num_heads}")

# トークンのリストを取得
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# 例として1層目の1ヘッド目の注意マップをプロット
layer = 0  # 1層目（0から数える）
head = 0   # 1ヘッド目（0から数える）
attention = attentions[layer][0, head].detach().cpu().numpy()

# 注意マップの保存
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(attention, cmap='viridis', aspect='auto')

# 軸にトークンを表示
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=90)
ax.set_yticklabels(tokens)

# カラーバーを追加
fig.colorbar(cax)

# タイトルとラベルを設定
plt.title(f'Attention Map - Layer {layer+1}, Head {head+1}')
plt.xlabel('Key Position')
plt.ylabel('Query Position')

# 画像を保存
plt.savefig(f'attention_map_layer_{layer+1}_head_{head+1}.png')
plt.close()

print(f"Attention map saved as attention_map_layer_{layer+1}_head_{head+1}.png")
