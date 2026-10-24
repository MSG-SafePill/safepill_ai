# SafePill AI

SafePill AI/OCR 파트입니다. 현재 구조는 Python FastAPI 추론 서비스와, 해당 서비스를 호출하기 위한 Java 클라이언트 실험 코드로 나뉩니다.

## 현재 역할

- YOLO 기반 알약 영역 탐지
- EasyOCR 기반 약 봉투/알약 문자 인식
- OCR 결과와 DB 약품 카탈로그 매칭
- 식별된 약품 기반 AI 상담

## 권장 실행 환경

- Python 3.11 권장
- PostgreSQL
- OpenAI API Key

현재 로컬에 Python 3.14가 잡혀 있다면 `torch`, `easyocr`, `ultralytics`, `numpy` 조합에서 호환성 문제가 날 수 있습니다. AI 담당자는 별도 Python 3.11 가상환경을 쓰는 것을 권장합니다.

## Python 서비스 설정

1. 가상환경 생성

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 의존성 설치

```powershell
pip install -r python_service\requirements.txt
```

3. 환경변수 설정

필요한 환경변수를 `.env` 또는 실행 셸에 설정합니다.

필수 항목:

- `OPENAI_API_KEY`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASS`

검출·분류 모델 경로는 아래 YOLO11 학습 결과가 기본값입니다. 다른 파일을 사용할 때만 `SAFEPILL_MODEL_PATH`, `SAFEPILL_PILL_CLS_MODEL_PATH`를 설정합니다.

개발 중 OpenAI 키 없이 상호작용 분석 API만 확인하려면:

```powershell
$env:SAFEPILL_INTERACTION_OFFLINE="true"
```

## Python API 서버 실행

```powershell
uvicorn python_service.app:app --host 0.0.0.0 --port 8000
```

발표에서는 단일 worker와 reload 없는 실행을 권장합니다. 모델은 시작 시 한 번 로딩하며, 실패한 모델의 API는 503을 반환합니다. `/health`는 기존과 동일한 프로세스 생존 확인이며 모델 준비 완료를 보장하지 않습니다. 시작 로그의 `startup_failed`를 확인하고 실제 이미지로 점검해야 합니다.

`run_server.ps1`은 기존 `.venv`를 사용합니다. 의존성 설치는 명시적으로 `-InstallDependencies`를 지정할 때만 수행하고, 개발용 자동 재시작은 `-Reload`로 선택합니다.

2026-10-21 당시 코드 분석, 변경 내역과 검증 한계: [AI 성능·안정성 점검 보고서](docs/AI_PERFORMANCE_REVIEW_2026-10-21.md). 이 보고서의 YOLOv8 기본 경로는 당시 상태이며, 현재 YOLO11 경로는 아래 설명을 따릅니다.

헬스 체크:

```powershell
curl http://localhost:8000/health
```

## YOLO11 새 학습

`train.py`와 `scripts/train_pill_classifier.py`는 YOLO11 구조 YAML에서 무작위 초기화하고 `pretrained=False`, `resume=False`로 학습합니다. 기존 YOLOv8 가중치나 YOLO11 사전학습 `.pt`를 입력으로 받지 않습니다. Ultralytics `8.3.148`에는 두 구조가 포함되어 있습니다. 아래 명령은 `safepill_ai` 폴더에서 실행합니다.

### 알약 영역 검출 (`/infer`, `/ocr`의 선택적 영역 검출, `/identify`)

`data.yaml`에 유효한 train/val 이미지와 대응하는 YOLO TXT 박스 라벨이 있어야 합니다. 현재 기본 `data.yaml`은 이 PC에 없는 `C:/AIData/pill_dataset`을 가리키므로 실제 데이터 경로로 수정하거나 `--data`로 다른 YAML을 지정해야 합니다. 기본 클래스가 `pill` 하나이므로 이 모델은 알약 **위치만** 검출하며 종류를 구별하지 않습니다.

```powershell
python train.py --data .\data.yaml --device 0
```

완료된 새 모델은 `models/pill_detect/yolo11n/best.pt`에 복사되며 서버의 기본 검출 경로도 이 파일입니다. 이미 파일이 있으면 덮어쓰지 않으므로 다른 `--output-model`을 지정하세요. 기존 `runs/detect/train/weights/best.pt`는 보존됩니다.

### 약품 종류 분류 (`/classify-pills`, 실제 Flutter 낱알 화면)

`datasets/pill_mvp/yolo_cls/train/<약품ID>/`, `val/<약품ID>/`에 독립적으로 촬영한 이미지가 필요합니다. 현재 매핑의 약품 ID는 `40792`, `29002`, `34342`입니다. 예제 사진 3장은 학습·검증 세트를 대신할 수 없습니다.

```powershell
python scripts/train_pill_classifier.py --check-only
python scripts/train_pill_classifier.py --device 0
```

완료된 새 모델은 `models/pill_cls/yolo11n/best.pt`에 복사되며 서버의 기본 분류 경로도 이 파일입니다. 두 새 가중치가 생성되기 전에는 해당 모델을 사용하는 API가 503을 반환할 수 있습니다. 학습 후 별도 test 이미지로 세 약품과 미지원 약품을 검증한 뒤 앱에서 사용하세요. 다른 경로를 사용할 때는 각각 `SAFEPILL_MODEL_PATH`, `SAFEPILL_PILL_CLS_MODEL_PATH`로 지정할 수 있습니다.

## 주요 엔드포인트

처방전 사진에서 약품명만 확인하려면 다음 명령을 사용합니다. 약품명 열을 읽는 방식이므로 다른 양식의 처방전에서는 결과를 반드시 확인하세요.

```powershell
.\.venv\Scripts\python.exe -m python_service.run_ocr --image "test\강민준_처방전.jpg" --medicines-only
```

앱의 처방전 OCR 업로드 API도 이 약품명 추출 경로를 사용합니다. OCR 결과는 자동 등록 전에 확인이 필요합니다.

- `GET /health`: 서비스 상태 확인
- `POST /infer`: YOLO 탐지 결과 반환
- `POST /ocr`: OCR 결과 반환
- `POST /identify`: YOLO + OCR + DB 매칭 결과 반환
- `POST /chat`: 식별된 약품 기반 AI 상담
- `POST /interaction/analyze`: 약품/영양제 목록과 백엔드 상호작용 룰 기반 AI 안전 요약
- `POST /medication-match`: OCR/식별 텍스트를 백엔드 DB 등록용 후보(`itemId`, `itemType`)로 매칭

처방전 OCR 응답의 `scheduleSuggestions`는 백엔드 `POST /api/schedules/{regId}`에 넘길 수 있는 후보값입니다. OCR 결과는 오인식 가능성이 있으므로 프론트에서 사용자가 확인한 뒤 등록해야 합니다.

`/identify`와 `/prescription-ocr` 응답은 등록 연동을 위해 DB 매칭 후보를 포함합니다.

- `identifiedPills[].itemId`
- `identifiedPills[].itemType`
- `identifiedPills[].manufacturer`
- `items[].matchCandidates[]`

약품명 매칭 요청 예시:

```json
{
  "keywords": ["타이레놀정", "오메가3"],
  "topK": 5
}
```

응답 예시:

```json
{
  "requestId": "...",
  "status": "ok",
  "results": [
    {
      "keyword": "타이레놀정",
      "candidates": [
        {
          "itemId": 1,
          "itemType": "MEDICINE",
          "itemName": "타이레놀정500mg",
          "manufacturer": "한국얀센",
          "score": 0.92
        }
      ]
    }
  ]
}
```

## 예시 요청

```json
{
  "imagePath": "C:/dev/safepill/sample/pill.jpg",
  "topK": 5
}
```

상호작용 분석 요청 예시:

```json
{
  "items": [
    {
      "itemName": "아스피린정",
      "itemType": "MEDICINE",
      "ingredients": [{"name": "aspirin"}],
      "efficacy": "해열, 진통",
      "precautions": "출혈 위험이 있는 환자는 주의"
    },
    {
      "itemName": "오메가3",
      "itemType": "SUPPLEMENT",
      "ingredients": [{"name": "omega-3"}],
      "precautions": "항응고제 복용자는 전문가 상담 권장"
    }
  ],
  "interactionRules": [
    {
      "itemNameA": "아스피린정",
      "itemNameB": "오메가3",
      "ingredientNameA": "aspirin",
      "ingredientNameB": "omega-3",
      "riskLevel": "CAUTION",
      "description": "출혈 경향이 증가할 수 있습니다."
    }
  ],
  "userProfile": {
    "age": 45,
    "conditions": ["고혈압"]
  }
}
```

## DB 주의사항

`python_service/db.py`는 현재 `pills`, `ingredients`, `pill_ingredients`, `warnings`, `interactions` 테이블을 사용합니다.

메인 백엔드의 `medicine_master`, `medicine_ingredient`, `interaction_rule` 구조와 아직 통합되어 있지 않습니다. MVP 1차 범위에서는 AI DB를 별도 테스트 데이터로 유지하고, 2차 단계에서 메인 백엔드 DB 구조와 매핑하거나 API 경유 방식으로 통합하는 것이 안전합니다.

## MVP 단계에서의 위치

1차 MVP에서는 AI/OCR을 필수 흐름에 넣지 않습니다.

우선순위:

1. Python 서비스 단독 실행 확인
2. 샘플 이미지로 `/infer`, `/ocr`, `/identify` 확인
3. 발표용 테스트 데이터 구성
4. 백엔드 본체와 연결 방식 결정
