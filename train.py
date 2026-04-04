from ultralytics import YOLO
import torch

def main():
    # 1. 그래픽카드(GPU) 인식 확인 (필수 체크!)
    print("==================================================")
    if torch.cuda.is_available():
        print(f"✅ GPU 인식 완료: {torch.cuda.get_device_name(0)}")
    else:
        print("🚨 경고: GPU를 찾을 수 없습니다. CPU로 학습하면 너무 오래 걸립니다!")
    print("==================================================\n")

    # 2. 실전용 Small 모델 불러오기 (앱 구동용 최적의 밸런스)
    model = YOLO('yolov8s.pt')

    print("🚀 [Safe Pill] 실전 AI 학습을 시작합니다! (모니터 끄고 푹 주무셔도 됩니다)")

    # 3. 영혼의 풀 트레이닝 설정
    results = model.train(
        data='data.yaml',         # 정답지와 사진 경로가 적힌 파일

        # [실전 핵심 파라미터]
        epochs=200,               # AI가 문제집을 최대 200번까지 반복해서 풉니다.
        patience=30,              # 30번 동안 성적이 안 오르면 미련 없이 조기 종료합니다.
        device=0,                 # 0번 GPU(RTX 4060)를 100% 사용하도록 강제 지정합니다.

        # [성능 및 속도 조절 파라미터]
        batch=16,                 # 4060의 VRAM(8GB)에 가장 안정적인 배치 사이즈입니다.
        # (만약 에러가 안 난다면 32로 올려서 속도를 더 높일 수 있습니다)
        imgsz=640,                # 사진 크기 (모바일 환경에 맞춘 기본값)
        workers=4,                # CPU가 GPU에게 데이터를 넘겨주는 속도 (Windows 권장값)

        # [결과물 정리 옵션]
        project='SafePill_AI',    # 결과를 저장할 최상위 폴더 이름
        name='yolov8s_full_run',  # 이번 학습 결과물이 저장될 폴더 이름 (예: SafePill_AI/yolov8s_full_run)

        plots=True,               # 학습 완료 후 예쁜 결과 그래프 자동 생성
        cache=False               # RAM 용량 초과 방지를 위해 캐시는 끕니다.
    )

    print("✨ 기나긴 학습이 드디어 완료되었습니다!")
    print("📂 앱에 넣을 최종 모델 파일은 [SafePill_AI/yolov8s_full_run/weights/best.pt] 에 있습니다.")

if __name__ == '__main__':
    # Windows 환경에서 멀티프로세싱(workers) 에러를 방지하기 위한 필수 구문입니다.
    main()