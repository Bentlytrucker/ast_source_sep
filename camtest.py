import cv2

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

# --- 추가: 원하는 해상도와 FPS 설정 ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
# ------------------------------------

if not cap.isOpened():
    print(f"오류: 카메라를 열 수 없습니다. (장치 인덱스: {CAMERA_INDEX})")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("오류: 프레임을 읽어올 수 없습니다.")
        break

    cv2.imshow('Camera Feed (1920x1080)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
