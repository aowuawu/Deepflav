import pandas as pd
from sklearn.metrics import classification_report
import argparse
import os

# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser(description="根据预测结果计算 Accuracy/Precision/Recall/F1 指标")
    parser.add_argument("--enzyme", type=str, default="MT", choices=["MT", "GTs", "ATs"],
                        help="酶家族类型 (默认: MT)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入的预测+标签对齐CSV文件 (默认: {enzyme}_test_predictions_with_labels.csv)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二分类阈值 (默认: 0.5)")
    
    args = parser.parse_args()
    
    # 设置默认文件名规则
    input_file = args.input if args.input else f"{args.enzyme}_test_predictions_with_labels.csv"
    output_csv = f"{args.enzyme}_performance_metrics.csv"
    
    # 标签名称映射
    target_names_map = {
        "MT": ["non_MT", "MT"],
        "GTs": ["non_GTs", "GTs"],
        "ATs": ["non_ATs", "ATs"]
    }
    target_names = target_names_map.get(args.enzyme, ["negative", "positive"])
    positive_label = args.enzyme  # 正样本类别名称
    
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到输入文件 '{input_file}'")
        print("   请先运行 evaluate_ROC_AUC.py 生成该文件。")
        return

    # === 1. 读取结果文件 ===
    print(f"📂 读取文件: {input_file}")
    df = pd.read_csv(input_file)

    # === 2. 确保真实标签是数值型 ===
    # 如果 label 列是字符串，自动转换；如果已经是数值则跳过
    if df['label'].dtype == 'object':
        unique_labels = df['label'].unique()
        if len(unique_labels) == 2:
            # 假设包含酶名称的为正样本(1)，另一个为负样本(0)
            pos_keyword = args.enzyme.lower()
            df['label'] = df['label'].apply(lambda x: 1 if pos_keyword in str(x).lower() else 0)
            print(f"   自动转换标签: {unique_labels} -> {{0, 1}}")

    # === 3. 将预测概率转换为预测标签 ===
    df['pred_label'] = df['score'].apply(lambda x: 1 if x > args.threshold else 0)

    # === 4. 提取真实标签和预测标签 ===
    y_true = df['label'].values
    y_pred = df['pred_label'].values

    # === 5. 计算所有指标 ===
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    # === 6. 提取关键指标 ===
    metrics = {
        'Metric': ['Accuracy', f'Precision ({positive_label})', f'Recall ({positive_label})', f'F1-score ({positive_label})'],
        'Value': [
            report['accuracy'],
            report[positive_label]['precision'],
            report[positive_label]['recall'],
            report[positive_label]['f1-score']
        ]
    }

    # === 7. 创建 DataFrame 并保存 ===
    metrics_df = pd.DataFrame(metrics)
    metrics_df['Value'] = metrics_df['Value'].round(4)  # 保留4位小数
    metrics_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ 指标已计算完成，并保存至: {output_csv}")
    print("\n" + "="*40)
    print(metrics_df.to_string(index=False))
    print("="*40)

if __name__ == "__main__":
    main()
