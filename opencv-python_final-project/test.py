import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==== 이미지 경로 설정 ====
empty_path = "images/empty.png"  # 비어있는(기준) 주차장 이미지
current_path = "images/parking.png"  # 현재 상태의 주차장 이미지 (차량 포함 가능)

# ==== 이미지 불러오기 및 전처리 ====
empty = cv2.imread(empty_path)  # 기준 이미지
current = cv2.imread(current_path)  # 현재 상태 이미지

# 기준 이미지와 동일한 크기로 현재 이미지 리사이징 (정합을 맞추기 위해)
h, w = empty.shape[:2]
current = cv2.resize(current, (w, h))

# 컬러 이미지를 흑백으로 변환 (처리 효율 및 라인 추출 목적)
gray_empty = cv2.cvtColor(empty, cv2.COLOR_BGR2GRAY)
gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

# ==== 주차선 마스크 및 객체 마스크 생성 ====
# 기준 이미지에서 밝은 영역(흰 주차선)을 이진화
_, line_mask = cv2.threshold(gray_empty, 200, 255, cv2.THRESH_BINARY)

# 주차선 마스크를 팽창시켜 보다 확실하게 마스킹
line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8))

# 주차선 외 영역을 객체가 있을 수 있는 영역으로 설정
object_mask = cv2.bitwise_not(line_mask)

# ==== 주차 공간 찾기 ====
# 기준 이미지에서 주차선 기반으로 이진화하여 윤곽선 검출
_, binary = cv2.threshold(gray_empty, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 사각형이고 면적이 일정 이상인 것만 추려 주차공간으로 판단
filtered_boxes, areas = [], []
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    if len(approx) == 4 and area > 1000:
        x, y, w_box, h_box = cv2.boundingRect(approx)
        filtered_boxes.append((x, y, x + w_box, y + h_box))
        areas.append(area)

# 가장 큰 윤곽(주차장 외곽 등)은 제외
max_idx = np.argmax(areas)
parking_slots = [box for i, box in enumerate(filtered_boxes) if i != max_idx]

# 좌우 순서로 정렬 (ex: A1, A2, A3...)
parking_slots = sorted(parking_slots, key=lambda b: b[0])
slot_labels = ["A1", "A2"]  # 공간이 2개로 가정됨

# ==== 객체(차량/장애물 등) 검출 ====
# 현재 이미지와 기준 이미지의 차이 계산
diff = cv2.absdiff(gray_current, gray_empty)

# 객체가 있을 수 있는 영역만 필터링
masked_diff = cv2.bitwise_and(diff, object_mask)

# 변화가 큰 영역 이진화하여 객체 후보 추출
_, thresh = cv2.threshold(masked_diff, 30, 255, cv2.THRESH_BINARY)

# 노이즈 제거 및 연결된 구조로 만들기 위한 모폴로지 연산
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# 최종 객체 윤곽선 검출
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 검출된 객체 정보 저장
detected_objects = []
threshold_vehicle = 35000  # 차량으로 판단할 면적 임계값
threshold_noise = 500  # 너무 작은 노이즈는 제거
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > threshold_noise:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        detected_objects.append({
            'box': (x, y, x + w_box, y + h_box),
            'area': area,
            'center': (x + w_box // 2, y + h_box // 2)
        })

# ==== 주차 상태 분류를 위한 변수 초기화 ====
font = cv2.FONT_HERSHEY_SIMPLEX
threshold_dist = 35  # 주차공간 중심과 객체 중심 간 거리 허용 오차
output = current.copy()  # 최종 출력용 이미지
slot_statuses = [{'status': 'Empty', 'color': (0, 255, 0)} for _ in parking_slots]

# ==== 1단계: 각 객체가 어느 슬롯에 속하는지 판단 ====
for obj in detected_objects:
    obj_box = obj['box']
    max_overlap = 0
    home_slot_idx = -1
    # 주차 공간과의 겹치는 면적을 계산
    for i, slot_box in enumerate(parking_slots):
        ix1, iy1 = max(obj_box[0], slot_box[0]), max(obj_box[1], slot_box[1])
        ix2, iy2 = min(obj_box[2], slot_box[2]), min(obj_box[3], slot_box[3])
        overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if overlap > max_overlap:
            max_overlap = overlap
            home_slot_idx = i

    if home_slot_idx != -1:
        # 중심 거리로 위치 정확도 보정
        slot_cx = (parking_slots[home_slot_idx][0] + parking_slots[home_slot_idx][2]) // 2
        slot_cy = (parking_slots[home_slot_idx][1] + parking_slots[home_slot_idx][3]) // 2
        dist = np.sqrt((slot_cx - obj['center'][0]) ** 2 + (slot_cy - obj['center'][1]) ** 2)

        # 상태 결정: 차량/위반/장애물
        status_info = {}
        if obj['area'] >= threshold_vehicle:  # 차량으로 판단
            if dist > threshold_dist:
                status_info = {'status': 'Violation', 'color': (0, 0, 255), 'obj_info': obj}
            else:
                status_info = {'status': 'Occupied', 'color': (0, 255, 0), 'obj_info': obj}
        else:
            status_info = {'status': 'Obstacle', 'color': (0, 255, 255), 'obj_info': obj}

        slot_statuses[home_slot_idx] = status_info

# ==== 2단계: 차량이 다른 공간을 침범했는지 확인 ====
for i, slot_info in enumerate(slot_statuses):
    if 'obj_info' in slot_info and slot_info['status'] in ['Occupied', 'Violation']:
        main_obj_box = slot_info['obj_info']['box']
        for j, other_slot_box in enumerate(parking_slots):
            if i == j or slot_statuses[j]['status'] != 'Empty':
                continue
            ix1, iy1 = max(main_obj_box[0], other_slot_box[0]), max(main_obj_box[1], other_slot_box[1])
            ix2, iy2 = min(main_obj_box[2], other_slot_box[2]), min(main_obj_box[3], other_slot_box[3])
            if max(0, ix2 - ix1) * max(0, iy2 - iy1) > 0:
                # 침범된 공간은 사용할 수 없음
                slot_statuses[j] = {'status': 'Unavailable', 'color': (0, 255, 255)}

# ==== 3단계: 결과 시각화 ====
unavailable_count = 0
for i, status_info in enumerate(slot_statuses):
    x1, y1, x2, y2 = parking_slots[i]
    color = status_info['color']
    label = f"{slot_labels[i]} {status_info['status']}"

    # 콘솔출력
    if status_info['status'] in ['Obstacle', 'Violation']:
        print(f"{slot_labels[i]} : {status_info['status']}")

    # 주차 불가 공간 수 집계
    if status_info['status'] in ['Occupied', 'Obstacle', 'Violation', 'Unavailable']:
        unavailable_count += 1

    # 주차 공간 사각형 및 라벨 표시
    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    text_size = cv2.getTextSize(label, font, 0.7, 2)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    cv2.putText(output, label, (text_x, y1 - 15), font, 0.7, color, 2)

    # 객체 박스 및 라벨 표시
    if 'obj_info' in status_info:
        ox1, oy1, ox2, oy2 = status_info['obj_info']['box']
        status = status_info['status']

        box_color, text = (0, 0, 0), ""
        if status == 'Occupied':
            box_color, text = (0, 255, 0), "Vehicle"
        elif status == 'Obstacle':
            box_color, text = (0, 255, 255), "Obstacle"
        elif status == 'Violation':
            box_color, text = (0, 0, 255), "Violation"

        if text:
            cv2.rectangle(output, (ox1, oy1), (ox2, oy2), box_color, 2)
            cv2.putText(output, text, (ox1, oy1 - 10), font, 0.6, box_color, 2)

# 총 주차 불가 개수 출력
print(f"현재 {unavailable_count}칸 주차 불가")

# ==== 결과 출력 ====
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Parking Detection")
plt.axis("off")
plt.show()