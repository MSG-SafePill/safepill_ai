from ultralytics import YOLO

# 1. AI 모델 불러오기
# (yolov8n.pt는 '나노' 버전으로, 제일 가볍고 빨라서 첫 테스트용으로 최고입니다)
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    print("🚀 세이프필 AI 학습을 시작합니다!")

    # 2. 학습(Training) 시작 명령!
    results = model.train(
        data='data.yaml',  # 방금 만든 지도 파일 이름
        epochs=10,         # AI가 문제집을 총 몇 번 반복해서 풀 것인가? (일단 가볍게 10번)
        imgsz=640,         # 사진을 공부할 때 640x640 크기로 맞춰서 보기
        batch=16,          # 한 번에 16장씩 머릿속에 집어넣고 공부하기
        plots=True         # 공부 끝나고 결과 그래프도 예쁘게 그려주기
    )

    print("✨ n차 학습이 완료되었습니다!")