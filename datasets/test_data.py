import cv2
import os
import random
from pathlib import Path

def visualize_yolo_pose(data_path, split="train", num_samples=3):
    img_dir = Path(data_path) / "images" / split
    label_dir = Path(data_path) / "labels" / split
    
    # 랜덤하게 이미지 선택
    img_files = list(img_dir.glob("*.jpg"))
    samples = random.sample(img_files, min(num_samples, len(img_files)))

    for img_path in samples:
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = list(map(float, line.split()))
            cls_id = int(data[0])
            # Bbox: xc, yc, wn, hn -> pixel
            xc, yc, wn, hn = data[1:5]
            x1 = int((xc - wn/2) * w)
            y1 = int((yc - hn/2) * h)
            x2 = int((xc + wn/2) * w)
            y2 = int((yc + hn/2) * h)

            # 박스 그리기 (초록색)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {cls_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 키포인트 그리기 (빨간색 점)
            kpts = data[5:]
            for i in range(0, len(kpts), 3):
                kx, ky, kv = kpts[i], kpts[i+1], kpts[i+2]
                if kv > 0: # 가시성이 0보다 큰 경우만 그림
                    px, py = int(kx * w), int(ky * h)
                    cv2.circle(img, (px, py), 3, (0, 0, 255), -1)

        # 결과 확인 (창 닫으려면 아무 키나 누르세요)
        cv2.imshow(f"Visualization: {img_path.name}", img)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 데이터셋 경로 입력
    visualize_yolo_pose("/home/yang/PROJECT/model_finetuning/datasets/ap10k_yolo")