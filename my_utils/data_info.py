import json
from collections import Counter

def count_labels(file_path):
    label_counter = Counter()
    total_samples = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                total_samples += 1
                try:
                    sample = json.loads(line)
                    # 根据实际数据，此处假定标签存储在 "label" 键中
                    label = sample.get('label', None)
                    if label is not None:
                        label_counter[label] += 1
                except json.JSONDecodeError as e:
                    print(f"解析错误: {e}，内容: {line}")
    return label_counter, total_samples

if __name__ == '__main__':
    file_path = "/home/hjh/Desktop/BTIG_pycontrast/data/CIC-IDS-2017/CIC-IDS-2017.jsonl"
    # file_path = "data/ids2018/merged.jsonl"
    counter, total = count_labels(file_path)
    print("样本总数量:", total)
    print("各类别标签数量:")
    for label, count in sorted(counter.items(), key=lambda x: x[0]):
        print(f"Label: {label}, Count: {count}")
    
    # 计算Label '0' 和 剩余标签的比例
    count_zero = counter.get(0, 0) or counter.get("0", 0)
    count_others = total - count_zero

    if count_others > 0:
        ratio = count_zero / count_others
        # 格式化为 1:X.X
        if ratio > 1:
            print(f"\n比例: {ratio:.1f}:1")
        else:
            print(f"\n比例: 1:{(1/ratio):.1f}")
    else:
        print("\n比例: 无法计算")