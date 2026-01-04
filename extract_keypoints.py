import cv2
import pandas as pd
import os
from ultralytics import YOLO
from tqdm import tqdm

# 1. 설정
MODEL_PATH = 'runs/pose/AP10K_Pose/dog_cat_v1/weights/best.pt'  # 학습된 모델 경로
VIDEO_ROOT = './action_videos'  # 행동별 영상이 담긴 루트 폴더
OUTPUT_CSV = 'pet_action_dataset.csv'
IMG_SIZE = 640

# 2. 모델 로드
model = YOLO(MODEL_PATH)

def extract_keypoints():
    dataset = []
    
    # VIDEO_ROOT 내부의 각 폴더(행동 라벨) 순회
    # 예: action_videos/sit/video1.mp4 -> 라벨 'sit'
    actions = [d for d in os.listdir(VIDEO_ROOT) if os.path.isdir(os.path.join(VIDEO_ROOT, d))]
    
    for action in actions:
        action_dir = os.path.join(VIDEO_ROOT, action)
        video_files = [f for f in os.listdir(action_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"처리 중인 행동: {action} ({len(video_files)}개 영상)")
        
        for video_file in tqdm(video_files):
            video_path = os.path.join(action_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_cnt = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # YOLOv11 추론
                results = model.predict(frame, imgsz=IMG_SIZE, verbose=False, conf=0.5)
                
                for r in results:
                    # 가장 크게 검출된 객체 하나만 사용 (보통 영상엔 주인공 한 마리)
                    if len(r.keypoints) > 0:
                        # 정규화된 좌표(0~1) 추출 (x, y)
                        kpts = r.keypoints.xyn[0].cpu().numpy() 
                        
                        # 데이터를 한 줄로 펴기 (17개 관절 * 2좌표 = 34개 값)
                        flat_kpts = kpts.flatten().tolist()
                        
                        # [라벨, 영상파일명, 프레임번호, 좌표... ] 형태로 저장
                        row = [action, video_file, frame_cnt] + flat_kpts
                        dataset.append(row)
                
                frame_cnt += 1
            cap.release()

    # 3. CSV 저장
    # 컬럼명 생성 (x0, y0, x1, y1 ... x16, y16)
    columns = ['label', 'video_name', 'frame']
    for i in range(17):
        columns.extend([f'kpt_{i}_x', f'kpt_{i}_y'])
        
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"완료! 데이터셋이 {OUTPUT_CSV}에 저장되었습니다. 총 {len(df)}개 프레임 추출됨.")

if __name__ == "__main__":
    # action_videos 폴더 구조가 sit/, stand/, walk/ 등으로 되어있어야 합니다.
    if not os.path.exists(VIDEO_ROOT):
        os.makedirs(VIDEO_ROOT)
        print(f"'{VIDEO_ROOT}' 폴더를 생성했습니다. 여기에 행동별 폴더를 만들고 영상을 넣어주세요.")
    else:
        extract_keypoints()