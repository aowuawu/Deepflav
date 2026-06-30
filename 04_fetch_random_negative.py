import requests
import time
import csv
import random
import argparse

# ================= 参数设置 =================
TARGET_COUNT = 3200  # 最终需要的酶数量
POOL_SIZE = 4000     # 抓取的候选池大小
HEADERS = {"User-Agent": "RandomEnzymeDownloader/1.0 (your_email@example.com)"}

# ================= 辅助函数 =================
def fasta_to_csv(input_fasta, output_csv_file):
    """将 FASTA 文件转换为指定格式的 CSV 文件"""
    entries = []
    try:
        with open(input_fasta, "r", encoding="utf-8") as f:
            name = None
            sequence_lines = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if name and sequence_lines:
                        sequence = "".join(sequence_lines)
                        entries.append({"sequence": sequence, "label": 0, "name": name})
                    sequence_lines = []
                    try:
                        name = line.split("|")[1]
                    except IndexError:
                        name = line[1:].split()[0]
                else:
                    sequence_lines.append(line)
            
            if name and sequence_lines:
                sequence = "".join(sequence_lines)
                entries.append({"sequence": sequence, "label": 0, "name": name})

        with open(output_csv_file, "w", encoding="utf-8", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=["sequence", "label", "name"])
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)
                
        print(f"✅ CSV 转换完成：共生成 {len(entries)} 条记录，保存为 {output_csv_file}")
    except FileNotFoundError:
        print(f"❌ 文件未找到：{input_fasta}")

def download_and_convert(output_fasta, output_csv):
    """主流程：抓取候选池、随机抽样、下载FASTA、转为CSV"""
    
    # 第一步：抓取候选池
    print(f"========== 步骤 1：抓取约 {POOL_SIZE} 个非AT/MT/GT的酶候选池 ==========")
    # 增加排除 GT (Glycosyltransferase)
    query = 'reviewed:true AND (ec:*) AND NOT family:"Acyltransferase" AND NOT family:"Methyltransferase" AND NOT family:"Glycosyltransferase"'
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "fields": "accession",
        "format": "json",
        "size": 500
    }
    
    candidate_accessions = []
    while True:
        if len(candidate_accessions) >= POOL_SIZE:
            break
        response = requests.get(base_url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"查询失败：{response.status_code}")
            break
            
        data = response.json()
        results = data.get("results", [])
        for item in results:
            acc = item.get("primaryAccession")
            if acc:
                candidate_accessions.append(acc)
                
        print(f"已抓取 {len(candidate_accessions)} 个候选编号...", end="\r")
        
        next_link = response.links.get("next", {}).get("url")
        if not next_link:
            break
        base_url = next_link
        params = None
        time.sleep(1.0)
        
    print(f"\n候选池抓取完毕，共获取 {len(candidate_accessions)} 个编号。")

    # 第二步：随机打乱并截取
    print(f"========== 步骤 2：随机打乱并抽取 {TARGET_COUNT} 个编号 ==========")
    random.shuffle(candidate_accessions)
    selected_accessions = candidate_accessions[:TARGET_COUNT]
    print(f"已随机选取 {len(selected_accessions)} 个编号，准备下载序列。")

    # 第三步：批量下载 FASTA
    print(f"========== 步骤 3：下载 FASTA 序列 ==========")
    with open(output_fasta, "w", encoding="utf-8") as f_out:
        batch_size = 100
        for i in range(0, len(selected_accessions), batch_size):
            batch_ids = selected_accessions[i:i + batch_size]
            query_ids = " OR ".join(batch_ids)
            fasta_url = "https://rest.uniprot.org/uniprotkb/search"
            fasta_params = {
                "query": f"accession:({query_ids})",
                "format": "fasta",
                "size": batch_size
            }
            fasta_response = requests.get(fasta_url, headers=HEADERS, params=fasta_params)
            if fasta_response.status_code == 200:
                f_out.write(fasta_response.text)
            else:
                print(f"下载失败：{fasta_response.status_code}")
            time.sleep(1.0)
            
    print(f"✅ FASTA 下载完成，保存在 {output_fasta}")

    # 第四步：转化为 CSV
    print(f"========== 步骤 4：转换为 CSV ==========")
    fasta_to_csv(output_fasta, output_csv)

# ================= 运行入口 =================
def main():
    parser = argparse.ArgumentParser(description="从 UniProt 获取非 AT/MT/GT 的随机酶序列作为通用负样本")
    parser.add_argument("--enzyme", type=str, required=True, choices=["MT", "GTs", "ATs"],
                        help="当前训练的酶家族类型 (用于规范输出文件命名，查询逻辑本身是排除这三者)")
    
    args = parser.parse_args()
    
    output_fasta = f"{args.enzyme}_negative_enzymes_random.fasta"
    output_csv = f"{args.enzyme}_negative_enzymes_random.csv"
    
    print(f"🚀 开始为 {args.enzyme} 准备通用负样本 (随机蛋白)...")
    download_and_convert(output_fasta, output_csv)

if __name__ == "__main__":
    main()
