import requests
import time
import csv
import argparse
import os

# ================= 配置常量 =================
# 包含植物、动物、真菌、细菌的经典模式生物，富含非类黄酮的转移酶
SPECIES_LIST = {
    # --- 植物物种 ---
    "Arabidopsis thaliana": 3702,
    "Oryza sativa": 39947,
    "Nicotiana tabacum": 4097,
    "Zea mays": 4577,
    "Glycine max": 3847,
    "Medicago truncatula": 3880,
    "Solanum lycopersicum": 4081,
    "Vitis vinifera": 29760,
    "Populus trichocarpa": 3694,
    "Brachypodium distachyon": 15368,
    # --- 非植物物种 ---
    "Homo sapiens": 9606,
    "Mus musculus": 10090,
    "Danio rerio": 7955,
    "Drosophila melanogaster": 7227,
    "Caenorhabditis elegans": 6239,
    "Saccharomyces cerevisiae": 559292,
    "Escherichia coli": 83333
}

# 用于排除类黄酮相关的序列
FLAVONOID_KEYWORDS = ["flavonoid", "anthocyanin", "chalcone", "flavanone", "flavonol", "isoflavone"]

# UniProt 查询词映射
ENZYME_QUERY_MAP = {
    "MT": 'family:"Methyltransferase"',
    "ATs": 'family:"Acyltransferase"'
}

# 建议替换为你自己的邮箱，以符合 UniProt API 规范
HEADERS = {"User-Agent": "Enzyme_Downloader/1.0 (your_email@example.com)"}

# ================= 辅助函数 =================
def get_protein_desc(item):
    """稳健地提取蛋白描述字段，合并推荐名、替代名和提交名"""
    prot_desc = item.get("proteinDescription", {})
    parts = []
    rec = prot_desc.get("recommendedName", {}).get("fullName", {}).get("value")
    if rec:
        parts.append(rec)
    for alt in prot_desc.get("alternativeNames", []):
        val = alt.get("fullName", {}).get("value")
        if val:
            parts.append(val)
    for sub in prot_desc.get("submissionNames", []):
        val = sub.get("fullName", {}).get("value")
        if val:
            parts.append(val)
    return " ".join(parts).lower()

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

def download_and_convert(enzyme_type):
    """主流程：下载 FASTA 并转换为 CSV"""
    if enzyme_type not in ENZYME_QUERY_MAP:
        print(f"❌ 不支持的酶类型: {enzyme_type}")
        return

    query_family = ENZYME_QUERY_MAP[enzyme_type]
    output_fasta = f"noflav_{enzyme_type}_extended.fasta"
    output_csv = f"noflav_{enzyme_type}_extended.csv"

    print(f"========== 开始下载非类黄酮 {enzyme_type} ==========")
    all_accessions = []
    
    with open(output_fasta, "w", encoding="utf-8") as f_out:
        for species_name, taxon_id in SPECIES_LIST.items():
            print(f"查询 {species_name} 的 {enzyme_type}...")
            query = f'{query_family} AND organism_id:{taxon_id}'
            base_url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": query,
                "fields": "accession,id,protein_name,organism_name",
                "format": "json",
                "size": 500
            }
            
            species_accessions = []
            while True:
                response = requests.get(base_url, headers=HEADERS, params=params)
                if response.status_code != 200:
                    print(f"查询失败：{response.status_code}")
                    break
                    
                data = response.json()
                results = data.get("results", [])
                for item in results:
                    acc = item.get("primaryAccession")
                    if not acc:
                        continue
                    desc = get_protein_desc(item)
                    # 排除包含类黄酮关键词的蛋白
                    if not any(keyword in desc for keyword in FLAVONOID_KEYWORDS):
                        species_accessions.append(acc)
                
                # 处理分页
                next_link = response.links.get("next", {}).get("url")
                if not next_link:
                    break
                base_url = next_link
                params = None
                time.sleep(1.0)
                
            print(f" -> 从 {species_name} 获取了 {len(species_accessions)} 条非类黄酮 {enzyme_type} 序列")
            all_accessions.extend(species_accessions)

        # 统一按批次下载所有物种的 FASTA
        print(f"\n所有物种查询完毕，共找到 {len(all_accessions)} 条序列，正在批量下载 FASTA...")
        batch_size = 100
        for i in range(0, len(all_accessions), batch_size):
            batch_ids = all_accessions[i:i + batch_size]
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

    print(f"✅ FASTA 下载完成，保存在 {output_fasta}\n")
    
    print("========== 开始转换为 CSV ==========")
    fasta_to_csv(output_fasta, output_csv)

# ================= 运行入口 =================
def main():
    parser = argparse.ArgumentParser(description="从 UniProt 下载非类黄酮 MT/AT 序列并转为 CSV (负样本)")
    parser.add_argument("--enzyme", type=str, required=True, choices=["MT", "ATs"],
                        help="酶家族类型 (MT 或 ATs)")
    
    args = parser.parse_args()
    download_and_convert(args.enzyme)

if __name__ == "__main__":
    main()
