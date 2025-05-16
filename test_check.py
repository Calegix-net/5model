import torch
from transformers import DistilBertTokenizer, DistilBertModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# モデルとトークナイザーのロード
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased", output_attentions=True)

# ポジティブとネガティブの文章
text_pos = "This film effortlessly combines stunning visuals with a deeply moving narrative, leaving a lasting impact on the viewer."
text_neg = "The film drags on for far too long without offering anything new or interesting, making it incredibly dull and repetitive."

# トークナイズ（パディングとトランケーションを追加）
inputs_pos = tokenizer(text_pos, return_tensors="pt", padding=True, truncation=True)
inputs_neg = tokenizer(text_neg, return_tensors="pt", padding=True, truncation=True)

# モデルに入力してアテンションマップを取得
outputs_pos = model(**inputs_pos)
outputs_neg = model(**inputs_neg)

# アテンションマップの取得
attentions_pos = outputs_pos.attentions
attentions_neg = outputs_neg.attentions

# レイヤーとヘッドの数を取得
num_layers = len(attentions_pos)
num_heads = attentions_pos[0].size(1)

# アテンションマップの違いを統計的に分析
differences = np.zeros((num_layers, num_heads))

for layer in range(num_layers):
    for head in range(num_heads):
        # 最小のトークン数を取得し、その数に合わせてアテンションマップをスライス
        min_tokens = min(attentions_pos[layer][0, head].size(0), attentions_neg[layer][0, head].size(0))
        
        # スライスしたアテンションマップの差を計算
        attention_diff = torch.abs(attentions_pos[layer][0, head][:min_tokens, :min_tokens] - attentions_neg[layer][0, head][:min_tokens, :min_tokens]).mean().item()
        differences[layer, head] = attention_diff

# ディレクトリの作成
output_dir = "attention_maps_check"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 統計的な違いを可視化して、指定されたファイル名で保存
plt.figure(figsize=(12, 6))
sns.heatmap(differences, annot=True, cmap='viridis', xticklabels=[f'Head {i+1}' for i in range(num_heads)], yticklabels=[f'Layer {i+1}' for i in range(num_layers)])
plt.title("Mean Absolute Differences in Attention Maps")
plt.xlabel("Attention Heads")
plt.ylabel("Transformer Layers")
plt.savefig(os.path.join(output_dir, "attention_difference_heatmap.png"))
plt.close()