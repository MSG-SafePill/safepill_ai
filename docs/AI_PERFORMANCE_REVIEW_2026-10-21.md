# SafePill AI 성능·안정성 점검 — 2026-10-21

코드상 타임아웃 원인에 대한 수정과 환경 없이 가능한 회귀 검증을 완료했다. **실제 추론 성능 향상, 세 약품 인식 성공, 발표 환경의 타임아웃 해소는 아직 검증하지 못했다.** 학습 데이터와 분류 모델, 실행 패키지가 없어 YOLO 교체·재학습을 수행하지 않았다. 기존 모델·가중치·의존성 버전을 유지했다.

변경 전 세 저장소는 깨끗한 상태였다. 기준 커밋은 AI `c65c31694600cc16c9701195a076f0874c7552d7`, BE `f3a2fd8b6e694605f4a5dd1ae58fcf57574f5bd4`, FE `a2da7d30e17fd9ba8e60fd0ecf39cc522fefb0f1`이다. 커밋·푸시하지 않았다.

## 1. 기존 실제 처리 구조

| 경로 | 처리 |
|---|---|
| Flutter 낱알 화면 | `scan_medication.dart` → `VisionApi.identifyPill` → `ApiClient.postMultipart` |
| Spring 낱알 | `/api/vision/identify` → `VisionAnalysisService.identifyPill` → `PythonAiClient.classifyPills` |
| FastAPI 낱알 | `/classify-pills` → `MultiPillClassificationService.classify` → OpenCV 색상·윤곽 기반 영역 추출 → 영역별 JPEG 저장 → `PillClassificationService.classify` → YOLO **분류** |
| Spring 후처리 | 최상위 후보를 DB 약품·성분 정보와 매칭하여 기존 DTO 반환 |
| 처방전 | Flutter → `/api/ocr/prescription` → `/prescription-ocr-upload` → EasyOCR → 정규식 파서 → Python DB 매칭 → Spring DB 매칭 |
| 별도 기존 API | `/infer`: YOLO 검출. `/ocr`: 선택적 YOLO 영역 검출 + OCR. `/identify(-upload)`: YOLO 검출 + OCR + DB 순위화 |

실제 Flutter 낱알 경로는 `inference.py`의 검출 모델을 사용하지 않는다. 검출 모델만 YOLO11n/YOLO26n으로 바꿔도 앱의 3종 분류 속도가 개선된다고 볼 수 없다.

모델과 데이터:

- 검출: `YoloInferenceEngine.detect/infer`, 기본 `runs/detect/train/weights/best.pt`. `args.yaml`: YOLOv8n, 640px, 10 epochs, CPU 학습. 체크포인트를 실행하지 않고 ZIP/pickle opcode 메타데이터를 읽어 `DetectionModel`, 클래스 `{0: 'pill'}`, 저장 버전 **Ultralytics 8.4.22**를 확인했다.
- 분류: 기본 `models/pill_cls/best.pt` 또는 `SAFEPILL_PILL_CLS_MODEL_PATH`. **현재 로컬에 없음.** 기존 학습 스크립트 기본 초기 모델은 `yolov8n-cls.pt`이다. 실제 배포된 분류 모델의 구조·버전은 파일이 없어서 확인 불가다.
- 매핑: `40792` 디아셀캡슐50밀리그램, `34342` 써스펜데이정, `29002` 오피큐탄연질캡슐. 매핑 파일과 약품별 예제 이미지 3개는 있으나 학습 데이터는 아니다.
- `data.yaml`: `nc: 1`, `names: ['pill']`, 경로 `C:/AIData/pill_dataset/images/train`, `val`. 해당 디렉터리와 `datasets/pill_mvp/yolo_cls` 모두 없음. 검출 데이터와 분류 폴더 데이터는 서로 다르다.
- EasyOCR **1.7.2**, `['ko', 'en']`, 기본 CRAFT 텍스트 검출 + `korean_g2` 인식. 서버 기본 `SAFEPILL_OCR_GPU=false`. 전체 이미지 OCR은 최대 변 1800으로 축소 후 grayscale → bilateral filter → Otsu 이진화. 검출 영역은 padding 8px, 영역별 readtext 1회, 최종 문자열 중복 제거.
- 기존 YOLO 객체와 OCR pipeline은 `lru_cache`로 재사용, EasyOCR Reader도 캐시됨. **요청마다 모델을 새로 만드는 구조는 아니지만 첫 요청은 초기화 비용을 부담했다.**
- FastAPI/Uvicorn은 Docker에서 worker 1개, PowerShell 스크립트에서는 매 실행 pip 설치 및 `--reload`. AI 저장소의 Java 코드는 실험용 클라이언트이며 앱의 실제 백엔드는 `safepill_be`다.
- 로컬 RTX 4060 8GB 및 NVIDIA 드라이버는 확인. 그러나 PyTorch가 없으므로 CUDA 추론 가능 여부는 확인하지 못했다. CPU 학습 기록은 현재 서버가 CPU를 쓴다는 증거가 아니다. YOLO 장치는 `SAFEPILL_DEVICE`에 따르며 미설정 시 라이브러리가 선택한다.

## 2. 타임아웃 원인과 근거

운영 서버 로그/실행 환경이 없어 실제 장애 요청을 재현하지 못했다. 다음은 코드로 확인한 원인·위험 요인이며, 각 비중을 실측한 결과는 아니다.

1. **이벤트 루프 차단:** 업로드 엔드포인트가 `async def` 안에서 동기 YOLO/OCR/DB 작업을 직접 호출했다. 한 요청의 추론 중 다른 요청 처리도 지연될 수 있다. FastAPI는 일반 `def` 경로 함수는 thread pool로 실행하지만 직접 호출한 동기 함수는 자동으로 분리하지 않는다. [FastAPI 공식 설명](https://fastapi.tiangolo.com/async/#other-utility-functions)
2. **5초 전체 제한:** `PythonAiClient.postImage`가 업로드, 첫 모델 초기화, 추론, DB 응답을 모두 `.block(Duration.ofMillis(aiTimeoutMs))`로 대기했고 기본값은 5000ms였다. 실제 배포 환경에서 이 설정을 덮어쓰는지까지는 확인 불가다.
3. **첫 요청 초기화:** 캐시는 있으나 서버 시작 시 초기화하지 않아 첫 OCR 요청에서 Reader 로딩/가중치 다운로드까지 발생 가능했다.
4. **Flutter 무기한 대기:** multipart 전송과 응답 본문 읽기에 timeout이 없었다. 화면에는 `finally`가 있지만 Future가 끝나지 않으면 실행되지 않는다.
5. **불필요한 I/O·연산:** 영역마다 임시 JPEG 파일을 쓰고 다시 읽음. 동일한 OCR 영역이 중복되면 같은 readtext 반복. 화면에서 읽은 이미지를 서비스에서 다시 읽음. Python 카탈로그 전체 문자열을 키워드마다 정규화. 빈 OCR 결과도 DB 조회.
6. **DB/업로드:** Python 카탈로그 DB 연결·쿼리 제한이 없었고, 저장소의 실행 가능한 설정에 30MB multipart 제한이 없었다. 기본값/외부 설정에 의존했다.
7. **오류 전달:** WebClient 연결/시간 초과/상위 서버 오류를 인식 API용 `code/message`로 변환하는 처리가 없었다.

## 3. 수정 파일과 동작

파일명은 각 저장소 기준이다.

| 저장소 / 파일 | 변경 |
|---|---|
| AI `python_service/app.py` | 업로드 경로를 sync `def`로 전환하여 FastAPI worker thread에서 실행. lifespan에서 YOLO/Reader 초기화. 모델별 시작 실패 기록 후 해당 경로 503. 이미지 요청은 한 번에 1개, 겹치면 즉시 503 및 Retry-After. 기존 응답 모델 유지. |
| AI `vision_runtime.py` | 단계 시간, 요청 ID, ASGI 전체 응답 시간·수신 대기·바이트 기록. 업로드를 1MiB 단위 복사, 30MiB 검사, 실패한 임시 파일 정리. |
| AI `ocr_pipeline.py` | Reader 최초 생성 및 추론 lock. 동일한 padded 좌표 OCR 중복 제거. 로딩·디코딩·전처리·OCR 시간 기록. 기존 모델/해상도/필터/decoder 유지. |
| AI `image_io.py` | Windows 한글 파일 경로를 `np.fromfile + cv2.imdecode`로 처리, 손상 이미지 명시적 오류. |
| AI `inference.py`, `pill_classifier.py` | 공유 YOLO predict lock, 로딩/추론 호출 시간 기록. 분류 task·매핑 검증으로 일반 사전학습 라벨을 약품으로 사용하지 못하게 함. |
| AI `multi_pill_classifier.py` | 임시 JPEG 파일 대신 메모리 JPEG encode/decode. 기존 JPEG 기본 품질을 유지. 영역 검출·입력 디코딩·crop 준비 시간 기록. 실제 동일성은 OpenCV 실행 환경에서 추가 검증 필요. |
| AI `medication_matcher.py` | 카탈로그 정규화를 호출당 항목별 1회 수행. 기존 점수·순서 보존. 빈 키워드 DB 조회 생략. |
| AI `db.py` | 카탈로그 읽기 연결에만 connect 3초 / statement 10초 기본 제한. DB 데이터·스키마·SQL 변경 없음. 세션 옵션이며 DB 서버 전역 설정 변경 없음. |
| AI `run_server.ps1` | 기본 reload 제거, 기존 venv 사용. 패키지 설치는 `-InstallDependencies` 명시 시에만. |
| AI `scripts/train_pill_classifier.py` | train/val 사전 검사 및 기존 출력 가중치 덮어쓰기 방지. |
| BE `vision/client/PythonAiClient.java` | 이미지 전용 WebClient 재사용, 연결·읽기·전체 제한 분리, 자동 재시도 방지, 명확한 오류 및 시간 기록. 기존 chat/interaction 호출은 유지. |
| BE `VisionAiException`, `VisionExceptionHandler` | 연결 실패/준비 안 됨 503, 제한 초과 504, 파일 초과 413, 잘못된 입력 400, 처리/응답 오류 502를 `code/message`로 전달. |
| BE `VisionHttpConfiguration`, `vision-http.properties`, `VisionTimingFilter` | 모든 profile에서 기본 업로드 제한 및 이미지 HTTP 설정 적용, 요청 ID와 전체 backend 시간 기록. 외부 application 설정이 우선. |
| BE `VisionAnalysisService.java` | 기존 DB 매칭 로직을 유지하며 AI 호출 이후 후처리 시간 로그만 추가. |
| FE `api_client.dart` | 업로드+응답 본문을 하나의 90초 deadline으로 묶고 http 1.6의 abortable multipart로 시간 초과 시 요청 취소. 크기 검사, 한국어 오류, debug 전체 시간 로그. |
| FE `vision_api.dart`, `scan_medication.dart` | 이미 읽은 이미지 bytes 재사용, 업로드 전 용량 검사, 중복 분석 및 dispose 후 setState 방지. 기존 오류 결과 화면/로딩 종료 유지. |

대체약 추천·오남용 가이드·상호작용 분석 구현은 수정하지 않았다. DB 서버를 기동하거나 접근하지 않았고, 동기화 API 두 종류도 호출하지 않았다.

## 4. YOLO 선정

**이번 적용: 기존 YOLO 경로와 모든 가중치 유지.** 교체를 뒷받침할 학습 데이터, 분류 가중치, 실행 환경 및 실측 성능이 없기 때문이다.

| 후보 | 호환성 / 판단 |
|---|---|
| YOLO11n / YOLO11n-cls | 고정 버전 8.3.148에 YOLO11 분류 구성이 있음. 추후 분류 데이터 복원 후 먼저 비교할 후보. 실제 우수함을 확인한 것은 아님. |
| YOLO26n / YOLO26n-cls | 8.4.0에서 도입. 현재 8.3.148 pin에는 없음. 별도 검증 환경에서 의존성 업그레이드와 재학습 필요. |

근거: [8.3.148 YOLO11 분류 구성](https://github.com/ultralytics/ultralytics/blob/v8.3.148/ultralytics/cfg/models/11/yolo11-cls.yaml), [8.4.0 YOLO26 출시](https://github.com/ultralytics/ultralytics/releases/tag/v8.4.0). [YOLO26 공식 성능](https://docs.ultralytics.com/models/yolo26/)은 다른 데이터·장치의 결과이며 SafePill 실측값으로 사용하지 않았다.

추가 발견: 보관된 YOLOv8 검출 가중치 생성 버전 8.4.22와 requirements의 8.3.148이 다르다. 이 사실만으로 로딩 실패라고 단정할 수는 없지만 실제 로딩 회귀 확인이 필요하다. 확인 없이 pin을 변경하지 않았다. 실제 정상 동작하던 환경의 `pip freeze`, torch/CUDA 버전을 확보하는 것이 우선이다.

## 5. OCR 최적화 범위

모델/언어/해상도/전처리/인식 옵션은 그대로다. 서버 시작 시 Reader를 준비하고 재사용을 보장하며, 완전히 동일한 영역만 반복 실행을 생략한다. 서로 다른 작은 글자 영역을 합치거나 제거하지 않는다. 처방전은 기존처럼 전체 이미지를 사용한다.

GPU, 인식 batch 증가, 1800px보다 작은 축소, bilateral 제거, 다른 경량 모델은 **미적용**이다. 실제 텍스트·복약 정보 정확도 비교가 없기 때문이다. EasyOCR 1.7.2에서 CPU 경로는 `batch_size`만 키워도 GPU와 같은 배치 처리가 되지 않는다. [EasyOCR API](https://www.jaided.ai/easyocr/documentation/), [1.7.2 구현](https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/easyocr/easyocr.py)

Python에서 생성한 매칭 후보를 Spring이 다시 매칭하는 부분도 확인했지만, 응답 계약을 유지하기 위해 제거하지 않았다. 추가 병목이 확인되면 별도 작업으로 조정한다.

## 6. 최적화 전후 실제 측정

기준 커밋의 Python 소스를 임시 디렉터리로 추출한 뒤, 현재 소스와 동일한 측정 도구 및 동일한 이미지 파일로 실행 가능 여부를 확인했다. **추론은 한 번도 성공하지 않았으므로 평균·속도 향상률을 제시하지 않는다.**

| 측정 항목 | 기존 | 변경 후 | 이유 |
|---|---|---|---|
| YOLOv8 검출 호출 시간 | 측정 불가 | 측정 불가 | ultralytics/torch 미설치 |
| 낱알 분류 시간 | 측정 불가 | 측정 불가 | `models/pill_cls/best.pt` 없음 |
| OCR 처리 시간 | 측정 불가 | 측정 불가 | cv2/easyocr/torch 및 OCR 가중치 없음 |
| AI API 전체 응답 시간 | 측정 불가 | 측정 불가 | 실행 중인 8000/8080 서버 없음 |
| 디아셀·써스펜데이·오피큐탄 실제 인식 | 측정 불가 | 측정 불가 | 실제 분류 실행 불가 |
| 실제 이미지 요청 타임아웃 발생 여부 | 측정 불가 | 측정 불가 | 통합 환경 없음 |
| YOLO11n/YOLO26n 비교 | 측정 불가 | 측정 불가 | 학습 데이터/신규 학습 가중치 없음 |

기계 판독 결과: [검출 변경 전](validation/detection-baseline.json), [검출 변경 후](validation/detection-current.json), [분류 변경 전](validation/classification-baseline.json), [분류 변경 후](validation/classification-current.json), [OCR 변경 전](validation/ocr-baseline.json), [OCR 변경 후](validation/ocr-current.json). 이미지 SHA-256이 포함되어 있다. 이전 소스용 임시 디렉터리는 측정 가능 여부 확인 후 자동 정리했다.

## 7. 검증 결과

- Python `unittest`: **16개 통과, FastAPI 계약/동시성 테스트 5개 환경 부족으로 skip**. 파일 크기 경계·실패 시 정리, 요청 ID/로그, OCR Reader 동시 초기화 1회, 중복 영역 제거·다른 영역 유지, 분류 task/클래스 매핑, 매칭 점수·빈 입력을 검증했다.
- OCR/분류 테스트는 테스트 더블을 사용한 제어 흐름 테스트이다. 실제 OCR 정확도나 약품 인식 성공의 증거가 아니다.
- Python 소스 문법 컴파일 통과. FastAPI 기준 AST 비교: **32개 요청/응답 모델과 12개 경로의 method·인자·response_model 동일**. 실제 직렬화 통합 검증은 아직 못 했다.
- Java 이미지 클라이언트 테스트 5개와 Flutter 업로드 테스트 5개를 추가했으나 **JDK/Flutter SDK가 없어 실행 불가**. Java/Flutter 빌드 통과를 주장하지 않는다.
- 기존 가중치 3개 SHA-256이 변경 전과 동일. [API·가중치 확인 기록](validation/compatibility-audit.json)
- 각 저장소 `git diff --check` 통과. 실제 DB 접속·스키마 생성·데이터 수정 없음.

## 8. 남은 제한과 발표 전 필수 확인

1. 정상 동작하던 `models/pill_cls/best.pt`를 원본 환경에서 확보해 복사하거나 `SAFEPILL_PILL_CLS_MODEL_PATH`로 지정해야 한다. 3개 label과 매핑을 실제 `model.names`로 비교해야 한다. 검출용 `best.pt`를 이 자리에 넣으면 안 된다.
2. 기존 정상 Python 3.11 가상환경 또는 그 lock/freeze, CRAFT/korean_g2 가중치, 읽기 가능한 기존 DB 연결이 필요하다. 이번 작업 중 새 AI 패키지/모델은 설치·다운로드하지 않았다.
3. 모델 초기화는 서비스 시작 시간으로 이동했다. `/health`는 기존 liveness 응답이므로 성공하더라도 모델 파일 누락 여부와 실제 이미지 응답을 따로 확인해야 한다. 시작 실패 후 파일을 복원하면 프로세스를 재시작한다.
4. 초기 예산은 Spring connect 3초 / read 60초 / image total 70초, Flutter total 90초이다. **실측에 기반한 최종값이 아니라 검증용 기본값**이다. 준비된 모델로 여러 회 측정한 p95와 업로드 시간을 보고 조정한다. `safepill.ai.timeout-ms`는 기존 chat/interaction에 남아 있다.
5. 파일 상한은 30MiB(31,457,280 bytes), Spring multipart 전체 요청은 헤더 여유 포함 31MiB다. 외부 application 설정이나 프록시의 더 작은 제한은 별도 확인한다. FastAPI 파일 검사는 multipart 파싱 이후 적용된다.
6. 한 worker에서 동시 이미지 요청은 대기 대신 503이다. 단일 발표 사용자를 우선한 설정이며, 여러 명의 동시 사용 처리량을 개선했다는 뜻은 아니다. 여러 Uvicorn worker를 쓰면 모델 복사본·제한도 worker별이다.
7. 클라이언트 timeout은 진행 중인 native GPU/CPU 연산을 강제로 종료하지 못한다. 이미 실행된 추론은 완료될 때까지 슬롯을 유지하며 추가 요청을 거절한다. 드라이버/모델 자체 hang은 별도 프로세스 격리·재시작 전략이 필요한 미해결 문제다.
8. Spring 후처리 DB 조회가 길면 Flutter 90초 한도에서 끝나지만 backend DB 작업의 전역 제한까지 이번 변경에서 다루지 않았다. `medicine_enrichment`/`prescription_enrichment` 로그로 판단한다.
9. 이 PC에서 통합 서버는 시작하지 않았다. 기존 백엔드 profile에 `ddl-auto=update`가 있으므로, 검증 서버를 시작할 때는 기존 DB를 변경하지 않도록 `JPA_DDL_AUTO=none` 또는 검증 환경 정책을 명시해야 한다.

## 9. 이어서 실행할 검증 절차

AI 실행 환경이 준비된 저장소 루트에서:

```powershell
# 기존 정상 환경/학습 가중치의 실제 경로로 지정
$env:SAFEPILL_PILL_CLS_MODEL_PATH = 'C:\path\to\trained-pill-classifier\best.pt'
# CPU 기준선을 먼저 측정. GPU는 torch.cuda.is_available() 확인 후 별도로 비교.
$env:SAFEPILL_DEVICE = 'cpu'
$env:SAFEPILL_OCR_GPU = 'false'
.\run_server.ps1
```

다른 터미널에서 모델 로딩 로그를 확인한 뒤, 같은 이미지·가중치·device로 변경 전후 각각 실행한다. 다음 명령의 분류 가중치 경로는 실제 확보한 경로로 바꾼다. `--source-root`에 기준 커밋 소스 디렉터리를 지정하면 동일 도구로 이전 코드를 측정할 수 있다. 기존 환경을 유지하고 소스만 교체하여 비교한다.

```powershell
python scripts/benchmark_vision.py --mode detection --model runs/detect/train/weights/best.pt --images 'test/디아셀캡슐50밀리그램.png' 'test/써스펜데이정.png' 'test/오피큐탄연질캡슐 .png' --repeat 5 --output runs/benchmark/detection.json
python scripts/benchmark_vision.py --mode classification --model 'C:\path\to\trained-pill-classifier\best.pt' --images 'test/디아셀캡슐50밀리그램.png' 'test/써스펜데이정.png' 'test/오피큐탄연질캡슐 .png' --repeat 5 --output runs/benchmark/classification.json
python scripts/benchmark_vision.py --mode ocr --images 'test/강민준_처방전.jpg' --repeat 5 --output runs/benchmark/ocr.json
python scripts/benchmark_vision.py --mode api --api-url http://localhost:8000 --endpoint /classify-pills --images 'test/디아셀캡슐50밀리그램.png' 'test/써스펜데이정.png' 'test/오피큐탄연질캡슐 .png' --repeat 5 --output runs/benchmark/api-pill.json
python scripts/benchmark_vision.py --mode api --api-url http://localhost:8000 --endpoint /prescription-ocr-upload --images 'test/강민준_처방전.jpg' --repeat 5 --output runs/benchmark/api-ocr.json
python -m unittest discover -s tests -v
```

API 테스트에는 기존 AI 의존성과 `httpx`가 필요하다. 측정 도구는 패키지를 설치하거나 YOLO 모델을 자동 다운로드하지 않는다. OCR 가중치가 없을 때도 측정 불가로 종료한다. 첫 호출과 이후 5회 평균/최소/최대, 결과 hash를 구분해 기록한다. 검출/분류 시간은 전처리·후처리를 포함한 서비스 호출 wall time이며 순수 GPU 커널 시간과 다르다. GPU 측정 시 synchronize를 적용한다. 분류의 `expected_label_present_each_run`은 예상 클래스 존재 여부이며 의료적 정확도나 전체 데이터셋 정확도가 아니다.

Spring/Flutter에서는 DB 없는 단위 테스트부터 실행한다:

```powershell
# safepill_be, Java 17 환경
.\gradlew.bat test --tests 'com.meta.safepill_be.vision.client.PythonAiClientTest'
# safepill_fe, 프로젝트 Flutter SDK 환경
flutter test test/vision_timeout_test.dart
flutter analyze
```

마지막으로 실제 앱에서 세 약품·처방전, 비이미지/손상 파일, 30MiB 초과, AI 서버 중단, 지연 응답, 두 요청 동시 전송을 점검한다. 요청 ID로 `client_total`, `backend_total`, `ai_http_total`, `api_total`, `ocr_*`, `yolo_*`, DB 매칭 로그를 연결한다. Python `receive_wait_ms`는 ASGI 수신 대기 시간이며 순수 네트워크 전송 시간과 동일하지 않다. Flutter 로그는 debug 모드에만 출력한다.

추가 개선은 실제 로그가 확보된 뒤 OCR GPU 비교, CPU thread 수 튜닝, YOLO11n-cls 독립 재학습, GPU batch, ONNX/TensorRT 변환 순서로 검토할 수 있다. 학습 출력은 `--output-model models/pill_cls/yolo11n-comparison.pt`처럼 분리하고, 세 약품 및 미지원 약품 검증에서 기존보다 나빠지면 배포 모델을 변경하지 않는다.
