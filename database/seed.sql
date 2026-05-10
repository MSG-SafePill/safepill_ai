INSERT INTO pills (code, pill_name, manufacturer, dosage_form, shape, color, imprint_text)
VALUES
    ('TYLENOL-500', '타이레놀정500밀리그램', 'Janssen Korea', '정제', '원형', '흰색', 'TYLENOL500'),
    ('ASPIRIN-100', '아스피린프로텍트정100밀리그램', 'Bayer Korea', '장용정', '원형', '흰색', 'ASPIRIN100'),
    ('CIMETIDINE-200', '휴온스시메티딘정200밀리그램', 'Huons', '정제', '원형', '흰색', 'CIMETIDINE200')
ON CONFLICT (code) DO NOTHING;

INSERT INTO ingredients (name)
VALUES
    ('아세트아미노펜'),
    ('아스피린'),
    ('시메티딘')
ON CONFLICT (name) DO NOTHING;

INSERT INTO pill_ingredients (pill_id, ingredient_id, strength_text)
SELECT p.id, i.id, v.strength_text
FROM (
    VALUES
        ('TYLENOL-500', '아세트아미노펜', '500mg'),
        ('ASPIRIN-100', '아스피린', '100mg'),
        ('CIMETIDINE-200', '시메티딘', '200mg')
) AS v(code, ingredient_name, strength_text)
JOIN pills p ON p.code = v.code
JOIN ingredients i ON i.name = v.ingredient_name
ON CONFLICT (pill_id, ingredient_id) DO NOTHING;

INSERT INTO warnings (pill_id, warning_type, warning_text)
SELECT p.id, v.warning_type, v.warning_text
FROM (
    VALUES
        ('TYLENOL-500', 'MAX_DAILY_DOSE', '성인 기준 아세트아미노펜 1일 총량 4000mg을 초과하지 마세요.'),
        ('ASPIRIN-100', 'BLEEDING_RISK', '출혈성 질환, 수술 예정, 항응고제 복용 시 의료진 상담이 필요합니다.'),
        ('CIMETIDINE-200', 'RENAL_HEPATIC_CAUTION', '신장/간 기능 저하 환자는 복용 전 전문가 상담이 필요합니다.')
) AS v(code, warning_type, warning_text)
JOIN pills p ON p.code = v.code
WHERE NOT EXISTS (
    SELECT 1 FROM warnings w
    WHERE w.pill_id = p.id AND w.warning_type = v.warning_type
);

INSERT INTO interactions (pill_id, target_ingredient, severity, interaction_text)
SELECT p.id, v.target_ingredient, v.severity, v.interaction_text
FROM (
    VALUES
        ('ASPIRIN-100', '와파린', 'high', '항응고 효과가 과도해져 출혈 위험이 증가할 수 있습니다.'),
        ('CIMETIDINE-200', '테오필린', 'moderate', '혈중 농도 증가 가능성이 있어 용량 조절이 필요할 수 있습니다.')
) AS v(code, target_ingredient, severity, interaction_text)
JOIN pills p ON p.code = v.code
WHERE NOT EXISTS (
    SELECT 1 FROM interactions i
    WHERE i.pill_id = p.id
      AND i.target_ingredient = v.target_ingredient
      AND i.severity = v.severity
);
