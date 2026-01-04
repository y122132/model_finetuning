import json
import os
import shutil
from pathlib import Path

# --- 설정 (경로에 맞게 수정하세요) ---
base_path = Path("./ap-10k")
img_source_dir = base_path / "data"
ann_dir = base_path / "annotations"
output_dir = Path("./datasets/ap10k_yolo")

# 사용할 스플릿 정의 (Split1 기준)
splits = {
    "train": ann_dir / "ap10k-train-split1.json",
    "val": ann_dir / "ap10k-val-split1.json",
    "test": ann_dir / "ap10k-test-split1.json"
}

# 강아지(Dog), 고양이(Cat) 필터링 (필요 시 수정)
TARGET_CATEGORIES = {"dog": 0, "cat": 1}

def convert_to_yolo():
    for split_name, json_path in splits.items():
        print(f"처리 중: {split_name}...")
        
        # 출력 폴더 생성
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        with open(json_path, 'r') as f:
            data = json.load(f)

        # 카테고리 ID 매핑 (이름 기준)
        cat_id_map = {}
        for c in data['categories']:
            for name, new_id in TARGET_CATEGORIES.items():
                if name in c['name'].lower():
                    cat_id_map[c['id']] = new_id

        # 이미지 정보 인덱싱
        images = {img['id']: img for img in data['images']}

        # 어노테이션 처리
        for ann in data['annotations']:
            if ann['category_id'] not in cat_id_map:
                continue # 대상 동물이 아니면 제외

            img_info = images[ann['image_id']]
            img_filename = img_info['file_name']
            src_img_path = img_source_dir / img_filename
            
            if not src_img_path.exists():
                continue

            # 1. 이미지 복사
            shutil.copy(src_img_path, output_dir / "images" / split_name / img_filename)

            # 2. YOLO 라벨 생성
            img_w, img_h = img_info['width'], img_info['height']
            x, y, w, h = ann['bbox']
            
            # Bbox 정규화
            xc, yc, wn, hn = (x + w/2)/img_w, (y + h/2)/img_h, w/img_w, h/img_h
            
            # 키포인트 정규화 (17개)
            kpts = ann['keypoints']
            yolo_kpts = []
            for i in range(0, len(kpts), 3):
                yolo_kpts.extend([kpts[i]/img_w, kpts[i+1]/img_h, kpts[i+2]])

            # 텍스트 저장
            cls_id = cat_id_map[ann['category_id']]
            kpt_str = " ".join([f"{k:.6f}" for k in yolo_kpts])
            line = f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f} {kpt_str}\n"

            txt_name = Path(img_filename).stem + ".txt"
            with open(output_dir / "labels" / split_name / txt_name, 'a') as f_txt:
                f_txt.write(line)

    print(f"완료! 데이터가 {output_dir}에 저장되었습니다.")

if __name__ == "__main__":
    convert_to_yolo()