import cv2
import numpy as np


def process_hoof_image(input_path, output_path):
    img = cv2.imread(input_path)

    if img is None:
        return False, "Could not read image.", None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([40, 255, 255])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((15, 15), np.uint8)

    yellow_mask = cv2.GaussianBlur(yellow_mask, (9, 9), 0)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    yellow_contours = [c for c in contours if cv2.contourArea(c) > 500]

    if not yellow_contours:
        return False, "Calibration tape not found.", None, None

    contours, _ = cv2.findContours(
        yellow_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    print(f"Contours found: {len(contours)}")
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(i, area, w, h)

    if not contours:
        return False, "No hoof/boot shape detected.", None, None

    largest_contour = max(contours, key=cv2.contourArea)

    print("Largest contour area:", cv2.contourArea(largest_contour))

    x, y, w, h = cv2.boundingRect(largest_contour)
    print("x =", x)
    print("y =", y)
    print("w =", w)
    print("h =", h)
    print("image height =", img.shape[0])
    print("image width =", img.shape[1])
    print("Bounding width pixels:", w)
    print("Bounding height pixels:", h)
    print("Expected pixels/mm:", w / 140)
    img_h, img_w = img.shape[:2]

    print("Image size:", img_w, "x", img_h)
    print("Bounding box:", w, "x", h)

    if h > img_h or w > img_w:
        return False, "Bad contour detected. Try a clearer image.", None, None

    output_img = img.copy()

    cv2.drawContours(
        output_img,
        yellow_contours,
        -1,
        (0, 255, 255),
        4,
    )

    cv2.rectangle(
        output_img,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        4,
    )

    cv2.drawContours(
        output_img,
        [largest_contour],
        -1,
        (0, 0, 255),
        6,
    )

    cv2.imwrite(output_path, output_img)

    pixels_per_mm = w / 140
    measured_width = round(w / pixels_per_mm)
    measured_length = round(h / pixels_per_mm)

    tape_contour = max(yellow_contours, key=cv2.contourArea)

    tx, ty, tw, th = cv2.boundingRect(tape_contour)

    print("Tape width pixels:", tw)
    print("Tape height pixels:", th)

    measured_width = round(w / pixels_per_mm)
    measured_length = round(h / pixels_per_mm)

    message = f"Detected object box: {w}px wide x {h}px tall"

    return True, message, measured_length, measured_width
