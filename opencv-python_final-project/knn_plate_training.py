#knn_plate_training.py
import os
import numpy as np
from PIL import Image
import cv2

# 🔹 학습용 문자 이미지 디렉토리 설정
DIGIT_DIR = "data/result_digits"           # 숫자(0~9) 문자 이미지가 저장된 디렉토리
KOREAN_DIR = "data/result_korean"          # 한글 문자 이미지가 저장된 디렉토리
MODEL_PATH = "knn_model.yml"               # 학습된 KNN 모델을 저장할 파일 경로
LABEL_MAP_PATH = "label_map.txt"           # 인덱스 ↔ 문자 대응 정보를 저장할 파일 경로

CHAR_WIDTH, CHAR_HEIGHT = 20, 30           # 문자 이미지를 고정된 크기로 리사이즈 (너비 20, 높이 30)

# 🔹 PIL 이미지 객체를 OpenCV 배열(BGR 아님, 흑백)을 numpy 배열로 변환
def pil_to_cv2(pil_img):
    return np.array(pil_img)

# 🔹 문자 이미지들을 디렉토리에서 불러와 학습용 데이터와 라벨로 변환
def load_images_from_folder(folder, start_label_idx=0):
    data = []          # 학습용 이미지 벡터 리스트
    labels = []        # 각 이미지에 대응하는 라벨 인덱스 리스트
    label_map = {}     # 인덱스 ↔ 문자 이름 매핑 딕셔너리
    current_idx = start_label_idx  # 현재 라벨 인덱스 시작값

    # 라벨(문자) 별 디렉토리 순회 (예: 0, 1, 2, ..., 가, 나, 다 등)
    for label_name in sorted(os.listdir(folder)):
        label_path = os.path.join(folder, label_name)
        if not os.path.isdir(label_path):
            continue  # 디렉토리가 아닌 경우 건너뛰기

        label_map[current_idx] = label_name  # 인덱스 ↔ 문자 매핑 등록

        # 각 문자 폴더 내 이미지 순회
        for fname in os.listdir(label_path):
            img_path = os.path.join(label_path, fname)
            try:
                pil_img = Image.open(img_path).convert("L")  # 흑백 변환
                pil_img = pil_img.resize((CHAR_WIDTH, CHAR_HEIGHT))  # 크기 통일
                img = pil_to_cv2(pil_img)  # numpy 배열로 변환
            except Exception as e:
                print(f"[⚠️] 이미지 로드 실패 → 삭제됨: {img_path} → {e}")
                try:
                    os.remove(img_path)  # 오류 있는 파일 삭제 시도
                except:
                    pass
                continue

            data.append(img.flatten())        # 이미지를 1차원 벡터로 변환 후 추가
            labels.append(current_idx)        # 해당 문자 인덱스 라벨로 추가

        current_idx += 1  # 다음 문자 라벨 인덱스로 증가

    return np.array(data, dtype=np.float32), np.array(labels), label_map  # 학습 데이터, 라벨, 라벨맵 반환

# 🔹 인덱스 ↔ 문자 대응 라벨맵을 텍스트 파일로 저장
def save_label_map(label_map, path):
    with open(path, "w", encoding="utf-8") as f:
        for idx, label in label_map.items():
            f.write(f"{idx},{label}\n")  # 예: 0,가 / 1,나 ...
    print(f"[💾] 라벨맵 저장 완료: {path}")

# 🔹 OpenCV KNN 객체를 생성하고 학습 수행
def train_knn(data, labels):
    knn = cv2.ml.KNearest_create()                   # KNN 객체 생성
    knn.train(data, cv2.ml.ROW_SAMPLE, labels)       # 학습: 각 행이 하나의 샘플
    return knn

# 🔹 전체 파이프라인 실행 메인 함수
def main():
    print("[📂] 숫자 이미지 로딩 중...")
    digit_data, digit_labels, digit_map = load_images_from_folder(DIGIT_DIR, start_label_idx=0)
    # 숫자 이미지 로딩 및 인덱스 라벨 부여 (0번부터 시작)

    print("[📂] 한글 문자 이미지 로딩 중...")
    korean_data, korean_labels, korean_map = load_images_from_folder(KOREAN_DIR, start_label_idx=len(digit_map))
    # 한글 이미지 로딩 및 숫자 이후 인덱스부터 시작

    # 데이터 부족 예외 처리
    if len(digit_data) == 0 or len(korean_data) == 0:
        print("[❌] 학습할 데이터가 부족합니다. 이미지 생성 상태를 확인하세요.")
        return

    # ▶ 숫자 + 한글 데이터 통합
    train_data = np.vstack([digit_data, korean_data])              # 이미지 데이터 수직 병합
    train_labels = np.concatenate([digit_labels, korean_labels])   # 라벨도 연결
    label_map = {**digit_map, **korean_map}                         # 라벨맵도 병합

    print(f"[📊] 총 학습 샘플 수: {len(train_labels)}개 / 라벨 수: {len(label_map)}개")

    # ▶ KNN 학습
    knn = train_knn(train_data, train_labels)

    # ▶ 모델 저장
    knn.save(MODEL_PATH)
    print(f"[✅] 모델 저장 완료: {MODEL_PATH}")

    # ▶ 라벨맵 저장
    save_label_map(label_map, LABEL_MAP_PATH)

# 🔹 스크립트 직접 실행 시 main() 실행
if __name__ == "__main__":
    main()