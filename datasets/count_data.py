import os
from collections import Counter


def count_dataset_classes(base_path):
    splits = ['train', 'val', 'test']
    print(f"{'Split':<10} | {'Dog (0)':<10} | {'Cat (1)':<10} | {'Total':<10}")
    print("-" * 50)

    for split in splits:
        label_dir = os.path.join(base_path, 'labels', split)
        if not os.path.exists(label_dir):
            continue

        counts = Counter()
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

        for file in label_files:
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    cls_id = line.split()[0]
                    counts[cls_id] += 1
        
        dog_count = counts.get('0', 0)
        cat_count = counts.get('1', 0)
        total = dog_count + cat_count
        
        print(f"{split:<10} | {dog_count:<10} | {cat_count:<10} | {total:<10}")

if __name__ == "__main__":
    # 데이터셋 경로가 정확한지 확인하세요
    dataset_path = "/home/yang/PROJECT/model_finetuning/datasets/ap10k_yolo"
    count_dataset_classes(dataset_path)