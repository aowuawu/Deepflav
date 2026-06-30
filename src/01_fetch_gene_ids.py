import requests
from time import sleep
import re
import argparse
import os

# =========================
# 读取EC文件
# =========================
def load_ec_file(ec_file):
    ec_list = []
    with open(ec_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and line.count('.') == 3:
                ec_list.append(line)
    return ec_list

# =========================
# 获取EC详细信息
# =========================
def get_ec_entry(ec):
    url = f"https://rest.kegg.jp/get/ec:{ec}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"获取 EC:{ec} 失败: {e}")
        return None

# =========================
# 解析GENES字段
# =========================
def parse_genes_block(text):
    lines = text.split("\n")
    genes_data = {}
    capture = False
    for line in lines:
        if line.startswith("GENES"):
            capture = True
            line = line[12:] # 去掉"GENES "
        elif capture:
            if not line.startswith(" "):
                break
            line = line.strip()
        else:
            continue
        
        if ":" in line:
            species, genes_str = line.split(":", 1)
            species = species.strip().lower()
            genes = []
            for g in genes_str.strip().split():
                g = re.sub(r"\(.*?\)", "", g)
                genes.append(g)
            genes_data.setdefault(species, []).extend(genes)
    return genes_data

# =========================
# 生成基因ID文件
# =========================
def generate_enzyme_genes(ec_list, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, ec in enumerate(ec_list, 1):
            print(f"[{i}/{len(ec_list)}] Processing EC {ec}")
            text = get_ec_entry(ec)
            f.write(f"ec:{ec}\n")
            
            if not text:
                f.write("No genes found\n\n")
                continue
                
            genes_dict = parse_genes_block(text)
            if not genes_dict:
                f.write("No genes found\n\n")
                continue
                
            for species, gene_ids in genes_dict.items():
                f.write(f"spe:{species}\n")
                for gid in gene_ids:
                    f.write(gid + "\n")
            f.write("\n")
            sleep(1) # 控制访问频率，防止被KEGG封IP

# =========================
# 主程序入口
# =========================
def main():
    parser = argparse.ArgumentParser(description="从 KEGG 数据库获取指定酶家族的基因ID")
    parser.add_argument("--enzyme", type=str, default="MT", choices=["MT", "GTs", "ATs"],
                        help="酶家族类型 (默认: MT)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入的 EC 编号列表文件 (默认: {enzyme}_ec_list.txt)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出的基因ID文本文件 (默认: {enzyme}_enzyme_genes.txt)")
    
    args = parser.parse_args()
    
    # 设置默认文件名规则
    ec_file = args.input if args.input else f"{args.enzyme}_ec_list.txt"
    output_file = args.output if args.output else f"{args.enzyme}_enzyme_genes.txt"
    
    if not os.path.exists(ec_file):
        print(f"❌ 错误：找不到输入文件 '{ec_file}'，请检查路径或先准备该文件。")
        return

    ec_list = load_ec_file(ec_file)
    if not ec_list:
        print(f"⚠️ 警告：文件 '{ec_file}' 中未找到有效的EC编号。")
        return

    print(f"🚀 开始获取 {args.enzyme} 家族的基因ID，共 {len(ec_list)} 个EC编号...")
    generate_enzyme_genes(ec_list, output_file)
    print(f"✅ 完成！结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
