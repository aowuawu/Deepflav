import os
import re
import torch
import numpy as np
import pandas as pd
import argparse
from esm import pretrained
from torch import nn

# ===== 配置 =====
DEFAULT_ESM_MODEL_PATH = "./esm_models/esm2_t33_650M_UR50D.pt"

# ===== 数据预处理 =====
def sanitize_sequence(seq):
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', seq.upper())

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

# ===== 加载模型 =====
def load_models(esm_model_path, model_weights_path, feature_mean_path, feature_std_path, device):
    if not os.path.exists(esm_model_path):
        raise FileNotFoundError(f"找不到ESM模型权重: {esm_model_path}")
    if not os.path.exists(model_weights_path) or not os.path.exists(feature_mean_path) or not os.path.exists(feature_std_path):
        raise FileNotFoundError("❌ 找不到分类器权重或特征参数，请确认训练阶段已生成这些文件。")

    # 加载本地 ESM 模型
    esm_model, alphabet = pretrained.load_model_and_alphabet_local(esm_model_path)
    esm_model = esm_model.to(device)
    esm_model.eval()

    # 加载微调分类器
    clf = MLPClassifier(input_dim=1280)
    clf.load_state_dict(torch.load(model_weights_path, map_location=device))
    clf = clf.to(device)
    clf.eval()

    # 加载特征标准化参数
    feature_mean = np.load(feature_mean_path)
    feature_std = np.load(feature_std_path)
    batch_converter = alphabet.get_batch_converter()

    return esm_model, clf, batch_converter, feature_mean, feature_std

# ===== 预测函数 =====
def predict_csv(input_csv, output_csv, esm_model_path, model_weights_path, feature_mean_path, feature_std_path, device):
    print("🚀 正在加载模型...")
    esm_model, clf, batch_converter, mean, std = load_models(
        esm_model_path, model_weights_path, feature_mean_path, feature_std_path, device
    )

    print(f"📂 读取序列文件: {input_csv}")
    df = pd.read_csv(input_csv)
    if "seq_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("CSV文件必须包含列: 'seq_id', 'sequence'")

    sequences = [(sid, sanitize_sequence(seq)) for sid, seq in zip(df["seq_id"], df["sequence"])]
    if not sequences:
        print("⚠️ CSV中没有序列!")
        return

    print("🧠 开始提取特征与预测...")
    batch_size = 4 if str(device) == 'cpu' else 8
    embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        ids, seqs = zip(*batch)
        batch_for_converter = list(zip(ids, seqs))
        _, _, tokens = batch_converter(batch_for_converter)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[33], return_contacts=False)
            emb = results["representations"][33][:, 1:-1].mean(1).cpu().numpy()
            embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = (embeddings - mean) / std

    X_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = clf(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    # 保存预测结果
    pred_df = pd.DataFrame({
        "seq_id": df["seq_id"],
        "score": probs
    })
    pred_df.to_csv(output_csv, index=False)
    print(f"✅ 预测完成，保存文件: {output_csv}")

# ===== 主流程 =====
def main():
    parser = argparse.ArgumentParser(description="使用训练好的 ESM-2 + MLP 分类器对测试集进行预测")
    parser.add_argument("--enzyme", type=str, default="MT", choices=["MT", "GTs", "ATs"],
                        help="酶家族类型 (默认: MT)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入的测试序列CSV (默认: {enzyme}_test_sequences_for_model_filtered.csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出的预测概率CSV (默认: {enzyme}_test_predictions.csv)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="训练阶段生成的模型和参数所在的根目录 (默认: ./output)")
    parser.add_argument("--esm_model_path", type=str, default=DEFAULT_ESM_MODEL_PATH,
                        help=f"ESM-2 模型权重路径 (默认: {DEFAULT_ESM_MODEL_PATH})")
    
    args = parser.parse_args()

    # 自动推导默认路径
    input_csv = args.input if args.input else f"{args.enzyme}_test_sequences_for_model_filtered.csv"
    output_csv = args.output if args.output else f"{args.enzyme}_test_predictions.csv"
    
    # 训练脚本默认将模型保存在 ./output/{enzyme}/ 下
    model_dir = os.path.join(args.output_dir, args.enzyme)
    model_weights = os.path.join(model_dir, f"{args.enzyme}_esm2_650M_nn_classifier.pth")
    feature_mean = os.path.join(model_dir, f"{args.enzyme}_esm2_650M_feature_mean.npy")
    feature_std = os.path.join(model_dir, f"{args.enzyme}_esm2_650M_feature_std.npy")

    if not os.path.exists(input_csv):
        print(f"❌ 错误：找不到输入文件 '{input_csv}'")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")

    predict_csv(input_csv, output_csv, args.esm_model_path, model_weights, feature_mean, feature_std, device)

if __name__ == "__main__":
    main()
