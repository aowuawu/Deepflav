import os
import torch
import esm
import pandas as pd
import re
import sys
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import argparse

# ===== 配置 =====
# 使用相对路径，要求将ESM模型权重放在项目根目录的 esm_models/ 文件夹下
DEFAULT_ESM_MODEL_PATH = "./esm_models/esm2_t33_650M_UR50D.pt"

# ===== 数据预处理 =====
def sanitize_sequence(seq):
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', str(seq).upper())

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels if labels is not None else ["unlabeled"] * len(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (str(self.labels[idx]), str(self.sequences[idx]))

def collate_fn(batch):
    return batch

# ===== 模型加载 =====
def load_esm_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到ESM模型权重: {model_path}。请先下载并放入指定目录。")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    return model.to(device).eval(), alphabet

# ===== 特征提取 =====
def extract_features(dataloader, model, alphabet, device):
    batch_converter = alphabet.get_batch_converter()
    features, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_labels, seq_str_list, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            results = model(tokens, repr_layers=[33])
            emb = results["representations"][33][:, 1:-1].mean(1)
            features.append(emb.cpu())
            labels.extend([int(label) for label in batch_labels])
            torch.cuda.empty_cache()
    return torch.cat(features).numpy(), np.array(labels)

# ===== 定义分类神经网络 =====
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1280):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

# ===== 日志设置 =====
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    class DualOutput:
        def __init__(self, *outputs):
            self.outputs = outputs
        def write(self, msg):
            for output in self.outputs:
                output.write(msg)
        def flush(self):
            for output in self.outputs:
                output.flush()

    log_file_handle = open(log_file, 'w')
    sys.stdout = DualOutput(sys.stdout, log_file_handle)
    sys.stderr = DualOutput(sys.stderr, log_file_handle)
    print(f"日志文件已创建: {log_file}")
    return log_file_handle

# ===== 主流程 =====
def main():
    parser = argparse.ArgumentParser(description="使用 ESM-2 特征训练 MLP 分类器")
    parser.add_argument("--enzyme", type=str, default="MT", choices=["MT", "GTs", "ATs"],
                        help="酶家族类型 (默认: MT)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入的训练集CSV文件 (默认: {enzyme}_training_new_set.csv)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="模型和日志的输出目录 (默认: ./output/{enzyme})")
    parser.add_argument("--esm_model_path", type=str, default=DEFAULT_ESM_MODEL_PATH,
                        help=f"ESM-2 模型权重路径 (默认: {DEFAULT_ESM_MODEL_PATH})")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数 (默认: 30)")
    
    args = parser.parse_args()
    
    # 设置默认路径
    input_csv = args.input if args.input else f"{args.enzyme}_training_new_set.csv"
    output_dir = args.output_dir if args.output_dir else f"./output/{args.enzyme}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert str(device) == "cuda", "❌ 必须使用GPU运行此脚本！"

    log_handle = setup_logging(output_dir)
    try:
        print("\n" + "="*50)
        print(f"加载数据: {input_csv}")
        df = pd.read_csv(input_csv)
        assert 'sequence' in df.columns and 'label' in df.columns, "CSV必须包含'sequence'和'label'列"
        
        df['sequence'] = df['sequence'].astype(str).apply(sanitize_sequence)
        df['label'] = df['label'].astype(int)

        # 过滤超长序列
        before_count = len(df)
        df = df[df['sequence'].str.len() <= 1024].reset_index(drop=True)
        after_count = len(df)
        print(f"序列长度过滤 (<=1024): 过滤前 {before_count} 条, 过滤后 {after_count} 条")

        pos_count = sum(df['label']==1)
        neg_count = sum(df['label']==0)
        print(f"正样本: {pos_count}, 负样本: {neg_count}, 比例: 1:{neg_count/pos_count:.1f}")

        # 划分训练集和验证集 (75% 训练, 25% 验证)
        train_df, val_df = train_test_split(df, test_size=0.25, stratify=df['label'], random_state=42)

        train_dataset = SequenceDataset(train_df['sequence'].tolist(), train_df['label'].tolist())
        val_dataset = SequenceDataset(val_df['sequence'].tolist(), val_df['label'].tolist())

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

        print("\n" + "="*50)
        print(f"加载ESM模型: {args.esm_model_path}")
        model, alphabet = load_esm_model(args.esm_model_path, device)

        print("\n提取训练集特征...")
        train_features, train_labels = extract_features(train_loader, model, alphabet, device)
        print("训练特征形状:", train_features.shape)

        print("提取验证集特征...")
        val_features, val_labels = extract_features(val_loader, model, alphabet, device)
        print("验证特征形状:", val_features.shape)

        # 特征标准化
        mean = train_features.mean(axis=0)
        std = train_features.std(axis=0) + 1e-6
        train_features = (train_features - mean) / std
        val_features = (val_features - mean) / std

        # 转换为 tensor
        X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
        y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
        X_val = torch.tensor(val_features, dtype=torch.float32).to(device)
        y_val = torch.tensor(val_labels, dtype=torch.float32).to(device)

        # 定义模型
        clf = MLPClassifier(input_dim=train_features.shape[1]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(clf.parameters(), lr=1e-4)

        print("\n" + "="*50)
        print("开始神经网络训练...")
        best_val_acc = 0
        
        for epoch in range(1, args.epochs + 1):
            clf.train()
            optimizer.zero_grad()
            outputs = clf(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            clf.eval()
            with torch.no_grad():
                val_outputs = clf(X_val)
                val_preds = torch.sigmoid(val_outputs) > 0.5
                acc = (val_preds.int() == y_val.int()).float().mean().item()

            print(f"[Epoch {epoch}/{args.epochs}] Loss: {loss.item():.4f}, Val Accuracy: {acc:.4f}")

            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(clf.state_dict(), os.path.join(output_dir, f"{args.enzyme}_esm2_650M_nn_classifier.pth"))
                np.save(os.path.join(output_dir, f"{args.enzyme}_esm2_650M_feature_mean.npy"), mean)
                np.save(os.path.join(output_dir, f"{args.enzyme}_esm2_650M_feature_std.npy"), std)
                print(f"--> 保存当前最优模型 (Acc: {acc:.4f})")

        print(f"\n训练完成，最佳验证准确率: {best_val_acc:.4f}")
        print(f"模型与特征参数已保存至: {output_dir}")

    except Exception as e:
        print(f"发生错误: {str(e)}", file=sys.stderr)
        raise
    finally:
        log_handle.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()
