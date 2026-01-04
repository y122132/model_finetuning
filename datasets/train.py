from ultralytics import YOLO
import torch

def start_training():
    # 1. GPU 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("경고: GPU를 찾을 수 없습니다. 학습 속도가 매우 느릴 수 있습니다.")
        device = 'cpu'
    else:
        print(f"GPU 감지됨: {torch.cuda.get_device_name(0)}")
        device = 0 # 0번 GPU 사용

    # # 2. 사전 학습된 YOLOv11n-pose 모델 로드
    # model = YOLO('yolo11s-pose.pt')

    # # 3. 모델 학습 시작
    # model.train(
    #     data='/home/yang/PROJECT/model_finetuning/datasets/data.yaml', # data.yaml 경로
    #     epochs=200,             # 학습 횟수
    #     patience=30,            # 20회 연속 성능 향상 없으면 조기 종료
    #     imgsz=640,              # 이미지 크기

    #     save=True,              # 가중치 저장
    #     project='AP10K_Pose',   # 결과 저장 폴더명
    #     name='dog_cat_v2',      # 실험 이름
    #     exist_ok=True,          # 기존 폴더 덮어쓰기 허용

    #     # --- 하드웨어 최적화 핵심 설정 ---
    #     batch=8,               # 배치 크기 (메모리 부족 시 8로 줄이세요)
    #     device=device,          # GPU/CPU 설정
    #     workers=2,              # 데이터 로딩용 CPU 코어 수
    #     amp=True,               # RTX 30 시리즈 필수: Mixed Precision으로 속도 향상 및 메모리 절약

    #     # --- 학습 알고리즘 최적화 ---
    #     optimizer='AdamW',      # 포즈 추정 및 파인튜닝에 가장 안정적인 옵티마이저
    #     lr0=0.01,               # 초기 학습률
    #     lrf=0.01,               # 최종 학습률 비율

    #     # --- 핵심 증강 옵션 최적화 ---
    #     fliplr=0.5,          # 좌우 반전 (필수: 데이터 2배 증폭 및 flip_idx 활성)
    #     flipud=0.1,          # 상하 반전 (조금만 적용: 특이 포즈 대비)
    #     mosaic=1.0,          # 배경 패턴 고착화 방지
    #     mixup=0.15,          # 배경/객체 분리 능력 향상
    #     copy_paste=0.3,      # 배경 무시 (배경 독립성 확보)
    #     degrees=30.0,        # 회전 증강 (동물의 다양한 신체 각도 대응)
        
    #     # --- 환경 적응력 향상 ---
    #     hsv_s=0.7,           # 배경 색상 변화 내성
    #     hsv_v=0.4,           # 조명 변화(낮/밤) 내성
    #     scale=0.5,           # 멀리 있거나 가까이 있는 동물 대비 (추가 권장)
    #     translate=0.1,        # 이동 증강 (추가 권장)

    #     pose=15.0,  # 포즈 정확도에 더 높은 배점을 부여
    #     kobj=2.0,   # 키포인트 존재 여부 판단 강화
    # )

    # 3. 모델 재학습

    # 1. 기존에 가장 성적이 좋았던 모델을 로드합니다.
    model = YOLO('/home/yang/PROJECT/model_finetuning/datasets/AP10K_Pose/dog_cat_v2/weights/best.pt')
    model.train(
        data='/home/yang/PROJECT/model_finetuning/datasets/data.yaml', # data.yaml 경로
        epochs=200,             # 학습 횟수
        patience=30,            # 20회 연속 성능 향상 없으면 조기 종료
        imgsz=640,              # 이미지 크기

        save=True,              # 가중치 저장
        project='AP10K_Pose',   # 결과 저장 폴더명
        name='dog_cat_v2_refine',      # 실험 이름
        exist_ok=True,          # 기존 폴더 덮어쓰기 허용

        # --- 하드웨어 최적화 핵심 설정 ---
        batch=8,               # 배치 크기 (메모리 부족 시 8로 줄이세요)
        device=device,          # GPU/CPU 설정
        workers=2,              # 데이터 로딩용 CPU 코어 수
        amp=True,               # RTX 30 시리즈 필수: Mixed Precision으로 속도 향상 및 메모리 절약

        # --- 학습 알고리즘 최적화 ---
        optimizer='AdamW',      # 포즈 추정 및 파인튜닝에 가장 안정적인 옵티마이저
        lr0=0.001,               # 초기 학습률
        lrf=0.01,               # 최종 학습률 비율

        # --- 포즈 정확도(Pose mAP)를 올리기 위한 팁 ---
        pose=15.0,               # 포즈 손실(loss)의 가중치를 살짝 높여 관절 위치에 더 집중하게 함
        kobj=2.0,                # 키포인트가 있는지 없는지 판단하는 가중치 상향

        # --- 핵심 증강 옵션 최적화 ---
        fliplr=0.5,          # 좌우 반전 (필수: 데이터 2배 증폭 및 flip_idx 활성)
        flipud=0.1,          # 상하 반전 (조금만 적용: 특이 포즈 대비)
        mosaic=1.0,          # 배경 패턴 고착화 방지
        mixup=0.15,          # 배경/객체 분리 능력 향상
        copy_paste=0.3,      # 배경 무시 (배경 독립성 확보)
        degrees=30.0,        # 회전 증강 (동물의 다양한 신체 각도 대응)
        
        # --- 환경 적응력 향상 ---
        hsv_s=0.7,           # 배경 색상 변화 내성
        hsv_v=0.4,           # 조명 변화(낮/밤) 내성
        scale=0.5,           # 멀리 있거나 가까이 있는 동물 대비 (추가 권장)
        translate=0.1        # 이동 증강 (추가 권장)
    )

if __name__ == "__main__":
    start_training()