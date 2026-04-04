import os
import random
import shutil
from tqdm import tqdm

# ==========================================================
# 1. 경로 설정 (탐색기 주소와 똑같은지 눈으로 한 번 꼭 확인하세요!)
# ==========================================================
# [원본 사진 폴더]
img_dir = r"C:\AIData\166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터\01.데이터\1.Training\원천데이터\경구약제조합_5000종"

# [우리가 방금 만든 정답지 폴더]
txt_dir = r"C:\AIData\labels_txt"

# [최종 완성될 YOLO 전용 데이터셋 폴더]
base_dir = r"C:\AIData\pill_dataset"
# ==========================================================

# 2. YOLO 전용 방 만들기 (train / val)
for split in ['train', 'val']:
    os.makedirs(os.path.join(base_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', split), exist_ok=True)

# 3. 폴더 끝까지 파고들어서 사진 파일(png, jpg) 싹 다 찾기!
all_imgs = []
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            all_imgs.append(os.path.join(root, file))

print(f"총 {len(all_imgs)}개의 알약 사진을 찾았습니다!")

# 4. 사진과 정답지(TXT) 짝꿍 맞추기
valid_pairs = []
for img_path in all_imgs:
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0] # 확장자 뗀 이름 (예: 알약1)
    txt_path = os.path.join(txt_dir, base_name + '.txt') # 짝꿍 정답지 경로

    # 짝꿍 정답지가 존재할 때만 리스트에 넣기
    if os.path.exists(txt_path):
        valid_pairs.append((img_path, txt_path, filename, base_name + '.txt'))

print(f"정답지와 완벽하게 짝이 맞는 데이터는 총 {len(valid_pairs)}개 입니다. 섞기 시작합니다!")

# 5. 순서를 무작위로 섞고 8:2 비율로 쪼개기
random.shuffle(valid_pairs)
split_idx = int(len(valid_pairs) * 0.8)

train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]

print(f"Train(공부용): {len(train_pairs)}개 / Val(시험용): {len(val_pairs)}개로 나눕니다.")

# 6. 파일 복사하는 함수
def copy_data(pairs, split_name):
    for img_src, txt_src, img_name, txt_name in tqdm(pairs, desc=f"{split_name} 복사 중"):
        # 사진 복사
        shutil.copy(img_src, os.path.join(base_dir, 'images', split_name, img_name))
        # 정답지 복사
        shutil.copy(txt_src, os.path.join(base_dir, 'labels', split_name, txt_name))

# 7. 진짜 실행!
copy_data(train_pairs, 'train')
copy_data(val_pairs, 'val')

print(f"\n✨ 대성공! {base_dir} 폴더에 YOLO를 위한 완벽한 데이터셋이 준비되었습니다.")
