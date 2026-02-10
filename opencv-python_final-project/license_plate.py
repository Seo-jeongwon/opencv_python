# license_plate.py
import cv2
import numpy as np
from PIL import Image
import os  # Import the os module for directory operations

# 🔧 문자 이미지 크기 및 경로 설정
CHAR_WIDTH, CHAR_HEIGHT = 20, 30  # KNN 학습 및 예측 시 사용할 문자 이미지 크기 (가로 x 세로)
LABEL_MAP_PATH = "label_map.txt"  # 문자 라벨 인덱스와 실제 문자를 매핑한 텍스트 파일
KNN_MODEL_PATH = "knn_model.yml"  # 학습된 OpenCV KNN 모델의 저장 파일 경로
DEBUG_CHARS_DIR = "debug_chars"  # 개별 문자 디버그 이미지 저장 경로
DEBUG_IMAGE_DIR = "debug_images"  # 중간 과정 이미지 저장 경로 추가


# 🔹 라벨맵 로드 함수
def load_label_map():
    """
    label_map.txt 파일을 읽어 문자 인덱스 ↔ 문자(label) 딕셔너리를 반환한다.
    예: {0: '가', 1: '나', ..., 35: '9'}
    """
    label_map = {}
    try:
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            for line in f:
                idx, label = line.strip().split(",")
                label_map[int(idx)] = label
    except FileNotFoundError:
        print(f"[❌] Error: {LABEL_MAP_PATH} 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        exit() # 파일이 없으면 프로그램 종료
    return label_map


# 🔹 KNN 모델 로드 함수
def load_knn_model():
    """
    저장된 학습 모델 파일(knn_model.yml)을 불러와 KNN 객체를 반환한다.
    """
    knn = cv2.ml.KNearest_create()
    try:
        knn = knn.load(KNN_MODEL_PATH)
    except Exception as e:
        print(f"[❌] Error loading KNN model: {KNN_MODEL_PATH} - {e}")
        print("KNN 모델 파일이 없거나 손상되었을 수 있습니다. 모델 학습이 필요합니다.")
        exit() # 모델 로드 실패 시 프로그램 종료
    return knn


# 🔹 이미지에서 번호판 영역을 검출하는 함수
def detect_plate(img):
    """
    원본 이미지에서 번호판으로 보이는 직사각형 영역을 검출한다.
    후보 영역 중 가장 가능성 높은 사각형을 선택해 크롭하여 반환한다.
    """
    # 디버그 이미지 저장 폴더 생성
    if not os.path.exists(DEBUG_IMAGE_DIR):
        os.makedirs(DEBUG_IMAGE_DIR)

    # 1. 전처리: 그레이스케일 변환 → 블러링 → 엣지 검출
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)  # 윤곽선 강조

    # 2. 디버깅용 시각화 출력 및 파일 저장
    cv2.imshow("1.gray", gray)
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "1_gray.png"), gray)
    cv2.imshow("2.blur", blur)
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "2_blur.png"), blur)
    cv2.imshow("3.edges", edges)
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "3_edges.png"), edges)

    # 3. 외곽선 탐색 (윤곽선 기반 사각형 후보 검출)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 가장 가능성 높은 번호판 후보 선택
    best_candidate = None
    best_score = 0
    img_center_x = img.shape[1] / 2  # 이미지 중심 x
    img_center_y = img.shape[0] * 0.75  # 아래쪽 중심 y (번호판은 하단에 위치 가능성이 높음)

    best_box = None  # best_box 초기화 추가

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h
        area = w * h

        # 위치 필터링: 너무 위/아래에 있으면 제외
        if not (img.shape[0] * 0.3 < y < img.shape[0] * 0.95):
            continue

        # 크기 및 비율 필터링: 번호판에 적합한 비율 대역 및 면적 조건
        if 2.5 < ratio < 7.0 and area > 3000 and w > 80 and h > 25:
            cx, cy = x + w / 2, y + h / 2
            dist = abs(cx - img_center_x) + abs(cy - img_center_y)  # 이미지 중심으로부터 거리
            score = area / (1 + dist)  # 중심에 가까우며 면적이 큰 것일수록 높은 점수

            if score > best_score:
                best_score = score
                best_candidate = img[y:y + h, x:x + w]
                best_box = (x, y, w, h)

    # 5. 최종 후보 시각화 및 반환
    if best_candidate is not None:
        x, y, w, h = best_box
        # 원본 이미지에 검출된 번호판 사각형 그리기
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("4.Plate", best_candidate)
        cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "4_detected_plate_cropped.png"), best_candidate)
        cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "5_original_with_plate_rect.png"), img)  # 원본에 사각형 그려진 이미지 저장
        return best_candidate

    print("[❌] 번호판 후보를 찾지 못했습니다.")  # 후보를 찾지 못했을 때 메시지 추가
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "5_original_no_plate_found.png"), img)  # 번호판 못 찾았을 때 원본 이미지 저장
    return None


# 🔹 검춤된 번호판 이미지에서 문자를 인식하는 함수
def recognize_plate_chars(plate_img, knn, label_map, base_image_filename="unknown_image"):
    """
    번호판 이미지(plate_img)에서 문자들을 추출하여 학습된 KNN 모델로 인식 후 문자열로 반환한다.
    base_image_filename: 원본 이미지 파일명 (확장자 제외), 디버그 파일명에 사용된다.
    """
    result = ""  # 최종 인식된 결과 문자열
    draw_img = plate_img.copy()  # 시각화용 이미지 복사

    # 디버그 이미지 저장 폴더 생성 (개별 문자)
    if not os.path.exists(DEBUG_CHARS_DIR):
        os.makedirs(DEBUG_CHARS_DIR)

    # 디버그 이미지 저장 폴더 생성 (중간 과정)
    if not os.path.exists(DEBUG_IMAGE_DIR):
        os.makedirs(DEBUG_IMAGE_DIR)

    # 1. 흑백 변환
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("6.gray", gray)
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "6_plate_gray.png"), gray)

    # 2. 이진화 (문자 → 흰색, 배경 → 검정)
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("7.threshold", thresh)
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "7_plate_threshold.png"), thresh)

    # 3. 닫힘 연산 (노이즈 제거 + 문자 내부 구멍 채움)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("8.morph", morph)
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "8_plate_morphology.png"), morph)

    # 4. 윤곽선 탐색 (각 문자 후보 영역)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []  # 문자 후보 영역 좌표 저장 리스트

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / w if w != 0 else 0
        area = w * h

        # 문자로 판단될 조건 (비율, 크기)
        if not (0.1 < aspect < 10.0 and 10 <= h <= 100 and 3 <= w <= 80 and area > 100):
            continue
        rois.append((x, y, w, h))

    # 5. 라인 정렬 및 정제
    if len(rois) == 0:
        print("[⚠️] 인식할 문자를 찾지 못했습니다.")
        return ""
    average_y = np.mean([y + h // 2 for (_, y, _, h) in rois])  # 중심 y
    # 같은 라인에 있는 문자만 필터링 (번호판은 대개 한 줄로 되어있으므로)
    rois = [r for r in rois if abs((r[1] + r[3] // 2) - average_y) < 15]
    rois = sorted(rois, key=lambda r: r[0])  # 왼쪽 → 오른쪽 정렬

    # 6. 문자 인식 (KNN 예측)
    for i, (x, y, w, h) in enumerate(rois):
        roi = morph[y:y + h, x:x + w]
        pil_img = Image.fromarray(roi).resize((CHAR_WIDTH, CHAR_HEIGHT))  # 20x30 리사이즈
        sample = np.array(pil_img).reshape(1, -1).astype(np.float32)

        if sample.shape[1] != knn.getVarCount():
            print(f"[❌] 샘플 크기 불일치: expected {knn.getVarCount()}, got {sample.shape[1]}")
            continue

        # KNN 예측 수행 (k=1로 설정하여 가장 가까운 이웃만 사용)
        ret, result_id, _, _ = knn.findNearest(sample, k=1)
        label = label_map.get(int(result_id[0][0]), '?')  # 인식된 문자
        result += label

        # 디버그용 개별 문자 이미지 저장
        # 파일명 형식: [원본 이미지 파일명]_char_[인덱스]_[인식된 라벨].png
        debug_char_path = os.path.join(DEBUG_CHARS_DIR, f"{base_image_filename}_char_{i}.png")
        cv2.imwrite(debug_char_path, np.array(pil_img))

        # --- 디버그용: 각 문자 이미지를 팝업 창으로 보여주기 시작 ---
        debug_display_img = np.array(pil_img)  # PIL Image를 다시 NumPy 배열로 변환
        # 이미지가 흑백이므로 컬러로 변환하여 텍스트를 그릴 수 있게 합니다.
        if len(debug_display_img.shape) == 2:
            debug_display_img = cv2.cvtColor(debug_display_img, cv2.COLOR_GRAY2BGR)

        # 인식된 라벨을 이미지에 표시 (빨간색으로)
        cv2.putText(debug_display_img, label, (5, CHAR_HEIGHT - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(f"Debug Char {i}: {label}", debug_display_img)
        # --- 디버그용: 각 문자 이미지를 팝업 창으로 보여주기 끝 ---

        # 디버깅 및 시각화 (원본 번호판 이미지에 사각형 및 텍스트 표시)
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_pos_y = max(y - 10, 5)
        # 한글 텍스트를 OpenCV에 직접 표시하는 것은 복잡하므로, 디버그 메시지로 대체
        print(f"[DEBUG] Char {i} → ID: {int(result_id[0][0])} → {label}")

    # 최종 인식 결과 출력
    cv2.imshow("9.Final Result", draw_img)
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "9_final_recognized_plate.png"), draw_img)  # 최종 결과 이미지 저장
    print("[✅ 인식 결과]", result)
    return result


# 🔹 전체 실행을 담당하는 메인 함수
def main(image_path):
    """
    인식 전체 파이프라인을 실행하는 함수.
    1) 이미지 로드 → 2) 번호판 검출 → 3) 문자 인식 → 결과 출력
    """
    print("[🔍] 번호판 인식 시작:", image_path)
    img = cv2.imread(image_path)

    # 원본 이미지 파일명 추출 (확장자 제외)
    # 예: "images/test_plate_02.jpg" -> "test_plate_02"
    image_filename_without_ext = os.path.splitext(os.path.basename(image_path))[0]

    # 디버그 이미지 저장 폴더 생성 (원본 이미지 포함)
    if not os.path.exists(DEBUG_IMAGE_DIR):
        os.makedirs(DEBUG_IMAGE_DIR)

    # 원본 이미지 로드 및 표시
    if img is None:
        print(f"[❌] 이미지 로드 실패: {image_path}")
        return

    cv2.imshow("0.original", img)
    cv2.imwrite(os.path.join(DEBUG_IMAGE_DIR, "0_original_input.png"), img)


    # 번호판 검출
    # detect_plate 함수 내부에서 '5.Detected'와 '4.Plate'도 imshow 및 imwrite 됨
    plate_img = detect_plate(img)


    # 문자 인식
    if plate_img is not None:
        knn = load_knn_model()
        label_map = load_label_map()
        # 추출한 파일명을 recognize_plate_chars 함수로 전달
        recognize_plate_chars(plate_img, knn, label_map, image_filename_without_ext)
    else:
        print("[❌] 번호판 후보를 찾지 못했습니다. 문자 인식을 건너뜝니다.")

    # 모든 창이 닫힐 때까지 대기 후 종료
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 🔹 스크립트 단독 실행 시 실행
if __name__ == "__main__":
    main("images/test_plate_01.jpg") # 테스트할 이미지 경로를 여기에 지정하세요.