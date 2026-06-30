import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
import os

# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser(description="合并预测结果与真实标签，计算 AUC 并绘制 ROC 曲线")
    parser.add_argument("--enzyme", type=str, default="MT", choices=["MT", "GTs", "ATs"],
                        help="酶家族类型 (默认: MT)")
    parser.add_argument("--pred_csv", type=str, default=None,
                        help="预测概率CSV文件 (默认: {enzyme}_test_predictions.csv)")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="真实标签CSV文件 (默认: {enzyme}_test_labels.csv)")
    
    args = parser.parse_args()
    
    # 设置默认文件名规则
    pred_csv = args.pred_csv if args.pred_csv else f"{args.enzyme}_test_predictions.csv"
    label_csv = args.label_csv if args.label_csv else f"{args.enzyme}_test_labels.csv"
    
    output_csv = f"{args.enzyme}_test_predictions_with_labels.csv"
    output_roc_png = f"{args.enzyme}_ROC_curve.png"
    output_roc_pdf = f"{args.enzyme}_ROC_curve.pdf"
    output_roc_svg = f"{args.enzyme}_ROC_curve.svg"
    
    # === 检查文件是否存在 ===
    if not os.path.exists(pred_csv) or not os.path.exists(label_csv):
        missing = []
        if not os.path.exists(pred_csv):
            missing.append(pred_csv)
        if not os.path.exists(label_csv):
            missing.append(label_csv)
        print(f"❌ 错误：以下文件不存在: {', '.join(missing)}")
        print("   请先运行 predict_test_set.py 生成预测文件，并确认标签文件存在。")
        return
    
    # === 读取文件 ===
    print(f"📂 读取预测文件: {pred_csv}")
    print(f"📂 读取标签文件: {label_csv}")
    pred_df = pd.read_csv(pred_csv)
    label_df = pd.read_csv(label_csv)
    
    # === 对齐 seq_id ===
    eval_df = pd.merge(label_df, pred_df, on="seq_id")
    if eval_df.empty:
        raise ValueError("❌ 没有匹配的 seq_id，请检查两个文件的 seq_id 是否一致")
    
    # === 保存对齐后的表格 ===
    eval_df.to_csv(output_csv, index=False)
    print(f"✅ 对齐后的表格已保存: {output_csv}")
    
    # === 提取标签和预测概率 ===
    y_true = eval_df['label'].values
    y_score = eval_df['score'].values
    
    # === 计算 ROC/AUC ===
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f"🎯 AUC = {roc_auc:.4f}")
    
    # === 绘制 ROC 曲线 ===
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)  # 对角线
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {args.enzyme}', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # === 保存三种格式 ===
    plt.savefig(output_roc_png, dpi=300)
    plt.savefig(output_roc_pdf)
    plt.savefig(output_roc_svg)
    plt.show()
    
    print(f"✅ ROC曲线已保存为 PNG: {output_roc_png}")
    print(f"✅ ROC曲线已保存为 PDF: {output_roc_pdf}")
    print(f"✅ ROC曲线已保存为 SVG: {output_roc_svg}")

if __name__ == "__main__":
    main()
