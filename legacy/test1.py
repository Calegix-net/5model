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
text_pos = "The characters, cast in impossibly engineered circumstances, are fully estranged from reality."
text_neg = "The characters, cast in impossibly contrived situations, are totally estranged from reality."

# トークナイズとパディング
inputs_pos = tokenizer(text_pos, return_tensors="pt", padding='max_length', max_length=21)
inputs_neg = tokenizer(text_neg, return_tensors="pt", padding='max_length', max_length=21)

# モデルに入力してアテンションマップを取得
outputs_pos = model(**inputs_pos)
outputs_neg = model(**inputs_neg)

# アテンションマップの取得
attentions_pos = outputs_pos.attentions
attentions_neg = outputs_neg.attentions

# レイヤーとヘッドの数を取得
num_layers = len(attentions_pos)
num_heads = attentions_pos[0].size(1)

# ディレクトリの作成
output_dir = "attention_maps_2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# トークン化された入力を取得
tokens_pos = tokenizer.convert_ids_to_tokens(inputs_pos["input_ids"][0])
tokens_neg = tokenizer.convert_ids_to_tokens(inputs_neg["input_ids"][0])

# アテンションマップの可視化と保存
for layer in range(num_layers):
    for head in range(num_heads):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        sns.heatmap(attentions_pos[layer][0, head].detach().numpy(), ax=axes[0], cbar=True, xticklabels=tokens_pos, yticklabels=tokens_pos, cmap="viridis")
        sns.heatmap(attentions_neg[layer][0, head].detach().numpy(), ax=axes[1], cbar=True, xticklabels=tokens_neg, yticklabels=tokens_neg, cmap="viridis")
        
        axes[0].set_title(f"Positive - Layer {layer+1}, Head {head+1}")
        axes[1].set_title(f"Negative - Layer {layer+1}, Head {head+1}")
        
        plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(axes[0].get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(axes[1].get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"attention_map_layer_{layer+1}_head_{head+1}.png"))
        plt.close()

# アテンションマップの違いを統計的に分析
differences = np.zeros((num_layers, num_heads))

for layer in range(num_layers):
    for head in range(num_heads):
        attention_diff = torch.abs(attentions_pos[layer][0, head] - attentions_neg[layer][0, head]).mean().item()
        differences[layer, head] = attention_diff

# 統計的な違いを可視化
plt.figure(figsize=(12, 6))
sns.heatmap(differences, annot=True, cmap='viridis', xticklabels=[f'Head {i+1}' for i in range(num_heads)], yticklabels=[f'Layer {i+1}' for i in range(num_layers)])
plt.title("Mean Absolute Differences in Attention Maps")
plt.xlabel("Attention Heads")
plt.ylabel("Transformer Layers")
plt.savefig(os.path.join(output_dir, "attention_difference_heatmap.png"))
plt.close()
