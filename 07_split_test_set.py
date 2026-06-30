import pandas as pd
import argparse
import os

# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser(description="拆分测试集为序列表和标签表，用于后续独立预测与评估")
    parser.add_argument("--enzyme", type=str, default="MT", choices=["MT", "GTs", "ATs"],
                        help="酶家族类型 (默认: MT)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入的测试集CSV文件 (默认: {enzyme}_test_set.csv)")
    
    args = parser.parse_args()
    
    # 设置默认文件名规则
    input_file = args.input if args.input else f"{args.enzyme}_test_set.csv"
    seq_output = f"{args.enzyme}_test_sequences_for_model.csv"
    label_output = f"{args.enzyme}_test_labels.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到输入文件 '{input_file}'")
        return

    print(f"🚀 开始拆分测试集: {input_file}")
    
    # 读取数据
    df = pd.read_csv(input_file)

    # 检查原始列
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError("❌ 输入文件必须包含 'sequence' 和 'label' 两列")

    # 生成 seq_id (如 seq1, seq2, ...)
    df["seq_id"] = ["seq{}".format(i+1) for i in range(len(df))]

    # 拆分序列表（用于模型预测）
    seq_df = df[["seq_id", "sequence"]]
    seq_df.to_csv(seq_output, index=False)

    # 拆分标签表（用于 ROC/AUC）
    label_df = df[["seq_id", "label"]]
    label_df.to_csv(label_output, index=False)

    print(f"✅ 拆分完成！")
    print(f"  序列表已保存至: {seq_output}")
    print(f"  标签表已保存至: {label_output}")

if __name__ == "__main__":
    main()
