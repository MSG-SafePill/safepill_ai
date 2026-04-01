import json
import os
from tqdm import tqdm

json_dir = r"C:\AIData\166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터\01.데이터\1.Training\라벨링데이터\경구약제조합_5000종"
output_dir = r"C:\AIData\labels_txt"

os.makedirs(output_dir, exist_ok=True)

json_files = []
for root, dirs, files in os.walk(json_dir):
    for file in files:
        if file.endswith('.json'):
            json_files.append(os.path.join(root, file))

print(f"총 {len(json_files)}개의 정답지를 찾았습니다! 불량품 개수를 세면서 변환합니다...")

# ==========================================
# [추가된 부분] 불량 카운터 0에서 시작!
defective_count = 0
# ==========================================

for file_path in tqdm(json_files):
    filename = os.path.basename(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        img_w = data['images'][0]['width']
        img_h = data['images'][0]['height']

        txt_filename = filename.replace(".json", ".txt")

        with open(os.path.join(output_dir, txt_filename), 'w', encoding='utf-8') as out_f:
            for ann in data['annotations']:
                bbox = ann.get('bbox', [])

                # 만약 숫자가 4개가 아니면(불량이면)?
                if len(bbox) != 4:
                    defective_count += 1  # 카운터를 1개 올리고
                    continue              # 패스!

                x, y, w, h = bbox

                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                out_f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# ==========================================
# 변환이 다 끝나면 터미널에 결과 발표!
print(f"\n✨ 작업 완료! C:\AIData\labels_txt 폴더를 확인하세요.")
print(f"🚨 발견된 불량 알약(위치 누락) 데이터: 총 {defective_count}개")
# ==========================================