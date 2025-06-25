import cv2
import datetime

cam = cv2.VideoCapture(0)
ret, frame1 = cam.read()
ret, frame2 = cam.read()

filter_modes = ["Original", "NightVision", "Thermal", "Inverted"]
current_filter = 0
button_areas = []
sensitivity = 10000

display_width = 1280
display_height = 720


def apply_filter(frame, mode):
    if mode == "NightVision":
        return cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_SUMMER)
    elif mode == "Thermal":
        return cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    elif mode == "Inverted":
        return cv2.bitwise_not(frame)
    return frame

def draw_buttons(frame):
    global button_areas
    button_areas = []
    button_width = 150
    button_height = 30
    for i, mode in enumerate(filter_modes):
        x1 = i * button_width
        y1 = 40
        x2 = x1 + button_width
        y2 = y1 + button_height
        button_areas.append((x1, y1, x2, y2))
        color = (0, 255, 0) if i == current_filter else (80, 80, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, mode, (x1 + 10, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def draw_slider(frame, value, max_val):
    bar_x, bar_y, bar_w, bar_h = 10, 10, 400, 20
    filled_w = int((value / max_val) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 255), -1)
    cv2.putText(frame, f"Sensitivity: {value}", (bar_x + 420, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def handle_click(event, x, y, flags, param):
    global current_filter, sensitivity
    # Check filter button clicks
    for i, (x1, y1, x2, y2) in enumerate(button_areas):
        if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
            current_filter = i
    # Check slider area
    if event == cv2.EVENT_LBUTTONDOWN and 10 <= x <= 410 and 10 <= y <= 30:
        # Convert x-position into sensitivity value
        rel_x = x - 10
        sensitivity = int((rel_x / 400) * 30000)
        sensitivity = max(1, min(sensitivity, 30000))


cv2.namedWindow("Motion Detector")
cv2.setMouseCallback("Motion Detector", handle_click)

while cam.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    base = frame1.copy()
    for c in contours:
        if cv2.contourArea(c) < sensitivity:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(base, (x, y), (x + w, y + h), (0, 0, 255), 2)

    filtered = apply_filter(base, filter_modes[current_filter])

    draw_slider(filtered, sensitivity, 30000)
    draw_buttons(filtered)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(filtered, f"{filter_modes[current_filter]} | {timestamp}",
                (10, filtered.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    cv2.imshow("Motion Detector", filtered)
    frame1 = frame2
    ret, frame2 = cam.read()

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == 81:  # left arrow
        sensitivity = max(1, sensitivity - 1000)
    elif key == 83:  # right arrow
        sensitivity = min(30000, sensitivity + 1000)

cam.release()
cv2.destroyAllWindows()
