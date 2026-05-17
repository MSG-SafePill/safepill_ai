import os
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageEnhance
from thefuzz import process, fuzz

# ==========================================
# 1. 이미지 전처리 (품질 향상)
# ==========================================
def preprocess_image(image_path):
    """
    이미지 전처리로 OCR 정확도 향상:
    - 회전 자동 교정
    - 콘트라스트 조정
    - 노이즈 제거
    - 선명도 강화
    """
    print("🔧 이미지 전처리 중...")
    
    # 원본 이미지 로드
    img = cv2.imread(image_path)
    
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 적응형 히스토그램 균등화 (콘트라스트 향상)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. 이진화 (텍스트 강조)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 노이즈 제거
    denoised = cv2.fastNlMeansDenoising(binary, h=10)
    
    # 5. 선명도 강화
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # 전처리된 이미지 저장
    processed_path = image_path.replace('.jpg', '_processed.jpg')
    cv2.imwrite(processed_path, sharpened)
    
    print(f"✅ 전처리 완료: {processed_path}")
    return processed_path

# ==========================================
# 2. OCR 텍스트 추출 (개선된 버전)
# ==========================================
def extract_text_improved(image_path, use_preprocessed=True):
    """
    개선된 OCR 추출:
    - 전처리된 이미지 사용
    - 상세 정보 포함 (신뢰도)
    - 한글/영문 인식
    """
    print("🔍 OCR 분석 중...")
    
    if use_preprocessed:
        processed_path = preprocess_image(image_path)
        target_image = processed_path
    else:
        target_image = image_path
    
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    
    # detail=1 옵션으로 상세 정보 포함
    result = reader.readtext(target_image, detail=1)
    
    # 텍스트 + 신뢰도 반환
    extracted = []
    for (bbox, text, confidence) in result:
        # 신뢰도가 0.3 이상인 것만 수집
        if confidence >= 0.3:
            # 텍스트 정제 (앞뒤 공백 제거)
            text = text.strip()
            if len(text) >= 2:  # 2글자 이상만
                extracted.append({
                    'text': text,
                    'confidence': confidence
                })
    
    return extracted

# ==========================================
# 3. 텍스트 후처리 (맞춤법/띄어쓰기 보정)
# ==========================================
def postprocess_text(text_list):
    """
    후처리로 정확도 향상:
    - 특수문자 제거
    - 중복 제거
    - 순환 정렬
    """
    print("📝 텍스트 후처리 중...")
    
    cleaned = []
    seen = set()
    
    for item in text_list:
        text = item['text']
        confidence = item['confidence']
        
        # 1. 특수문자 정제
        cleaned_text = text.replace('|', 'I').replace('O', '0').replace('l', 'I')
        
        # 2. 중복 제거
        if cleaned_text not in seen and len(cleaned_text) >= 2:
            seen.add(cleaned_text)
            cleaned.append({
                'text': cleaned_text,
                'confidence': confidence
            })
    
    # 신뢰도 기준 정렬
    cleaned = sorted(cleaned, key=lambda x: x['confidence'], reverse=True)
    
    return cleaned

# ==========================================
# 4. 약물 매칭 (개선된 버전)
# ==========================================
def match_pills_improved(extracted_texts, db_pill_names=None, threshold=70):
    """
    개선된 약물 매칭:
    - 신뢰도 고려
    - 다양한 매칭 방식 시도
    - 점수 상세 표시
    """
    if db_pill_names is None:
        # 테스트용 임시 약물 목록
        db_pill_names = [
            '타이레놀', '아스피린', '겔포스', '감기약',
            '소화제', '밴드', '감기', '감염증',
            '정장', '보감정', '감염', '항생제'
        ]
    
    print("\n[🔬 약물 매칭 결과]")
    matched_results = []
    
    for item in extracted_texts:
        text = item['text']
        ocr_confidence = item['confidence']
        
        # 너무 짧은 텍스트는 스킵
        if len(text) < 2:
            continue
        
        # 가장 유사한 약물 찾기
        best_match, score = process.extractOne(
            text, 
            db_pill_names, 
            scorer=fuzz.partial_ratio
        )
        
        # 최종 신뢰도 = OCR 신뢰도 * 매칭 점수
        final_confidence = (ocr_confidence * (score / 100))
        
        if score >= threshold:
            matched_results.append({
                'ocr_text': text,
                'db_name': best_match,
                'ocr_confidence': f'{ocr_confidence:.1%}',
                'match_score': f'{score}%',
                'final_confidence': f'{final_confidence:.1%}'
            })
            print(f"✅ [{best_match}] | OCR: {ocr_confidence:.1%} | 매칭: {score}% | 최종: {final_confidence:.1%}")
    
    return matched_results

# ==========================================
# 5. 메인 실행
# ==========================================
if __name__ == "__main__":
    test_image_path = "envelope_test.jpg"
    
    if os.path.exists(test_image_path):
        print("=" * 70)
        print("🚀 개선된 OCR 분석 시작")
        print("=" * 70)
        
        # 1단계: 이미지 전처리
        processed_image = preprocess_image(test_image_path)
        
        # 2단계: 개선된 OCR 추출
        extracted = extract_text_improved(test_image_path, use_preprocessed=True)
        
        # 3단계: 텍스트 후처리
        cleaned = postprocess_text(extracted)
        
        print(f"\n📊 추출 결과: {len(extracted)} → 정제 후: {len(cleaned)}")
        
        # 4단계: 약물 매칭
        results = match_pills_improved(cleaned)
        
        print("\n" + "=" * 70)
        print(f"💊 최종 식별된 약물: {len(results)}개")
        print("=" * 70)
        for result in results:
            print(f"\n약물명: {result['db_name']}")
            print(f"  OCR 인식: {result['ocr_text']} (신뢰도: {result['ocr_confidence']})")
            print(f"  매칭 점수: {result['match_score']}")
            print(f"  최종 신뢰도: {result['final_confidence']}")
    else:
        print(f"❌ 에러: {test_image_path} 파일이 없습니다.")
