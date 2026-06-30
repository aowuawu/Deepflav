import pandas as pd
import re
import argparse
import os

# ================= 配置 =================
MAX_LEN = 1024  # ESM-2 模型推荐的最大序列长度

# ================= 辅助函数 =================
def sanitize_sequence(seq):
    """将非标准氨基酸替换为 X"""
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', seq.upper())

# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser(description="过滤超长序列（长度 > 1022），防止显存溢出")
    parser.add_argument("--enzyme", type=str, default="MT", choices=["MT", "GTs", "ATs"],
                        help="酶家族类型 (默认: MT)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入的测试集CSV文件 (默认: {enzyme}_test_sequences_for_model.csv)")
    parser.add_argument("--max_len", type=int, default=MAX_LEN,
                        help=f"最大序列长度阈值 (默认: {MAX_LEN})")
    
    args = parser.parse_args()
    
    # 设置默认文件名规则
    input_csv = args.input if args.input else f"{args.enzyme}_test_sequences_for_model.csv"
    
    # 根据输入文件名自动生成输出文件名
    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    output_csv = f"{base_name}_filtered.csv"
    filtered_csv = f"{base_name}_filtered_out.csv"
    
    if not os.path.exists(input_csv):
        print(f"❌ 错误：找不到输入文件 '{input_csv}'")
        return

    # 读取数据
    df = pd.read_csv(input_csv)
    if "seq_id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("CSV文件必须包含列: 'seq_id', 'sequence'")

    # 清洗序列：将非标准氨基酸替换为 X
    df["sequence"] = df["sequence"].astype(str).apply(sanitize_sequence)
    df["seq_length"] = df["sequence"].str.len()

    # 分离超长序列
    before_count = len(df)
    filtered_df = df[df["seq_length"] > args.max_len].copy()
    df = df[df["seq_length"] <= args.max_len].reset_index(drop=True)
    after_count = len(df)
    removed_count = len(filtered_df)

    print(f"序列长度过滤 (<= {args.max_len}):")
    print(f"  过滤前: {before_count} 条")
    print(f"  过滤后: {after_count} 条")
    print(f"  移除:   {removed_count} 条")

    # 保存被过滤的超长序列（便于审查）
    if removed_count > 0:
        filtered_df.to_csv(filtered_csv, index=False)
        print(f"\n被过滤的 {removed_count} 条超长序列已保存至: {filtered_csv}")
        for _, row in filtered_df.iterrows():
            print(f"  - seq_id: {row['seq_id']}, 长度: {int(row['seq_length'])}")
    else:
        print("没有超长序列被过滤。")

    # 保存过滤后的有效序列
    df = df.drop(columns=["seq_length"])
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 过滤后的序列已保存至: {output_csv}")

if __name__ == "__main__":
    main()
