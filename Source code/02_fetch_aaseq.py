import requests
import csv
from time import sleep
import argparse
import os

# =========================
# 解析 enzyme_genes.txt
# =========================
def parse_input_file(input_file):
    data = []
    current_ec = None
    current_species = None
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("ec:"):
                current_ec = line[3:]
                current_species = None
            elif line.startswith("spe:"):
                current_species = line[4:]
            else:
                gene_id = line
                if current_ec and current_species:
                    full_id = f"{current_species}:{gene_id}"
                    data.append((current_ec, full_id))
    return data

# =========================
# 获取蛋白序列
# =========================
def fetch_aaseq(full_id, timeout=20, retries=3):
    url = f"https://rest.kegg.jp/get/{full_id}/aaseq"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            text = r.text.strip()
            if text.startswith(">"):
                # 去掉FASTA头，只保留序列
                seq = "".join(text.split("\n")[1:])
                return seq
            return None
        except requests.exceptions.Timeout:
            sleep(2 ** attempt)
        except requests.exceptions.RequestException:
            return None
    return None

# =========================
# 主流程
# =========================
def generate_csv(input_file, output_csv):
    data = parse_input_file(input_file)
    if not data:
        print(f"⚠️ 警告：输入文件 '{input_file}' 中没有找到有效的基因ID。")
        return

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ecid", "geneid", "aaseq"])
        for i, (ec, full_id) in enumerate(data, 1):
            print(f"[{i}/{len(data)}] {full_id}")
            sleep(1) # 控制速率，防止被封
            seq = fetch_aaseq(full_id)
            if seq:
                writer.writerow([ec, full_id, seq])
            else:
                writer.writerow([ec, full_id, ""])

# =========================
# 运行入口
# =========================
def main():
    parser = argparse.ArgumentParser(description="从 KEGG 数据库根据基因ID获取蛋白序列")
    parser.add_argument("--enzyme", type=str, default="MT", choices=["MT", "GTs", "ATs"],
                        help="酶家族类型 (默认: MT)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入的基因ID文本文件 (默认: {enzyme}_enzyme_genes.txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出的蛋白序列CSV文件 (默认: {enzyme}_enzyme_proteins.csv)")
    
    args = parser.parse_args()
    
    # 设置默认文件名规则，与第一个脚本呼应
    input_file = args.input if args.input else f"{args.enzyme}_enzyme_genes.txt"
    output_csv = args.output if args.output else f"{args.enzyme}_enzyme_proteins.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到输入文件 '{input_file}'，请检查路径或先运行 fetch_gene_ids.py。")
        return

    print(f"🚀 开始从 {input_file} 获取蛋白序列...")
    generate_csv(input_file, output_csv)
    print(f"✅ 完成！结果已保存至: {output_csv}")

if __name__ == "__main__":
    main()
