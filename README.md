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

`.env.example`을 참고해 필요한 값을 설정합니다.

필수 항목:

- `OPENAI_API_KEY`
- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASS`
- `SAFEPILL_MODEL_PATH`

개발 중 OpenAI 키 없이 상호작용 분석 API만 확인하려면:

```powershell
$env:SAFEPILL_INTERACTION_OFFLINE="true"
```

## Python API 서버 실행

```powershell
uvicorn python_service.app:app --reload --port 8000
```

헬스 체크:

```powershell
curl http://localhost:8000/health
```

## 주요 엔드포인트

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

## YOLO 제품명 분류 학습 (다중 클래스)

알약 이미지와 YOLO 라벨(`.txt`)을 준비해 아래 순서로 학습할 수 있습니다.

1) 클래스 이름 파일 준비 (`classes.txt`, 한 줄에 한 클래스명)

```text
tylenol_500
brufen_200
...
```

2) 학습/검증 데이터셋 생성 + `data.yaml` 자동 생성

```powershell
python split_data.py `
  --images-dir C:\AIData\pill_images `
  --labels-dir C:\AIData\pill_labels `
  --output-dir C:\AIData\pill_dataset `
  --class-names-file C:\AIData\classes.txt
```

3) 학습 실행

```powershell
python train.py --data C:\AIData\pill_dataset\data.yaml --epochs 200 --batch 16
```

학습 완료 후 모델은 출력 로그의 `.../weights/best.pt` 경로에 생성됩니다.
