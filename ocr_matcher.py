import os
import easyocr
import psycopg2
from dotenv import load_dotenv
from thefuzz import process, fuzz

# ==========================================
# 1. PostgreSQL에서 약물 이름 목록 가져오기
# ==========================================
def get_pill_names_from_db():
    load_dotenv() # .env 파일에서 DB 정보 로드

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS")
    )

    try:
        with conn.cursor() as cur:
            # 테이블 이름과 컬럼명은 실제 환경에 맞게 수정하세요.
            cur.execute("SELECT pill_name FROM pills")
            rows = cur.fetchall()

            # [('타이레놀',), ('아스피린',)] 형태의 결과를 ['타이레놀', '아스피린'] 리스트로 변환
            pill_names = [row[0] for row in rows]
            return pill_names
    finally:
        conn.close()

# ==========================================
# 2. 약 봉투 이미지에서 텍스트 추출하기 (EasyOCR)
# ==========================================
def extract_text_from_image(image_path):
    print("🔍 약 봉투 이미지를 분석하는 중...")
    # GPU가 없다면 gpu=False로 설정 (속도는 조금 느려집니다)
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

    # detail=0 옵션으로 텍스트 문자열만 리스트 형태로 깔끔하게 뽑아옵니다.
    extracted_texts = reader.readtext(image_path, detail=0)
    return extracted_texts

# ==========================================
# 3. 추출된 단어와 DB 약물 이름 매칭하기 (핵심 알고리즘)
# ==========================================
def match_pills(extracted_texts, db_pill_names, threshold=75):
    matched_results = []

    print("\n[매칭 결과 분석]")
    for text in extracted_texts:
        # 노이즈(의미 없는 1글자짜리 오타나 특수문자 등) 필터링
        if len(text.strip()) < 2:
            continue

        # process.extractOne: 추출된 단어(text)와 DB 리스트 중 가장 유사한 1개를 찾음
        # fuzz.partial_ratio: "타이레놀정500mg" 안에 "타이레놀"이 포함되어 있으면 높은 점수를 줌
        best_match, score = process.extractOne(text, db_pill_names, scorer=fuzz.partial_ratio)

        # 유사도 점수가 우리가 설정한 기준점(threshold) 이상일 때만 유효한 약으로 인정
        if score >= threshold:
            matched_results.append({
                "ocr_text": text,         # 봉투에서 읽힌 원래 글자
                "db_name": best_match,    # DB에서 찾은 진짜 약 이름
                "confidence": score       # 일치율(점수)
            })
            print(f"✅ 인식 성공: '{text}' -> DB 매칭: [{best_match}] (유사도 {score}%)")
        else:
            # 약 이름이 아닌 병원 이름, 환자 이름, 복용법 등은 여기서 걸러집니다.
            pass

    return matched_results

# ==========================================
# 4. 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 테스트용 가짜 이미지 경로 (실제 약 봉투 이미지 파일명으로 변경하세요)
    test_image_path = "envelope_test.jpg"

    # 1. DB에서 기준 데이터 세팅
    print("데이터베이스 연결 중...")
    db_pills = get_pill_names_from_db()

    # 2. OCR 추출
    if os.path.exists(test_image_path):
        raw_texts = extract_text_from_image(test_image_path)
        print(f"\n📄 OCR로 읽어낸 원본 텍스트:\n{raw_texts}")

        # 3. 데이터 매칭
        final_pills = match_pills(raw_texts, db_pills, threshold=75)

        print("\n💊 최종 식별된 약물 목록:")
        for pill in final_pills:
            print(f"- {pill['db_name']}")

        # 💡 이 final_pills 리스트를 기존에 만든 '챗봇(qa_chain)'에 넘겨주면,
        # "식별된 약물들의 주의사항과 알러지 충돌 여부를 스케줄로 짜줘"가 완성됩니다!
    else:
        print(f"에러: {test_image_path} 파일이 없습니다. 봉투 사진을 폴더에 넣어주세요.")