import cv2
import numpy as np
import mss
import time

# 1. Define your region of interest (ROI) on the screen
# Coordinates: {'top': y, 'left': x, 'width': w, 'height': h}
boardpx = {"bottom": 600, "left": 600, "right": 1150, "top": 400}

def recognize_card(card_corner):
    """Matches a grayscale, processed card corner against saved templates."""
    if not RANKS_TEMPLATES or not SUITS_TEMPLATES:
        return "??"  # Templates not populated yet

    # Convert card corner to grayscale for matching
    gray_corner = cv2.cvtColor(card_corner, cv2.COLOR_BGR2GRAY)
    _, thresh_corner = cv2.threshold(gray_corner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 1. Identify Rank
    best_rank = "?"
    best_rank_score = -1
    for rank_name, rank_template in RANKS_TEMPLATES.items():
        # Resize template dynamically to match size differences if needed,
        # or guarantee your template and corner crops scale identically.
        h, w = rank_template.shape[:2]
        # Match using normalized cross-correlation
        res = cv2.matchTemplate(thresh_corner, rank_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_rank_score and max_val > 0.70:  # 70% confidence threshold
            best_rank_score = max_val
            best_rank = rank_name

    # 2. Identify Suit
    best_suit = "?"
    best_suit_score = -1
    for suit_name, suit_template in SUITS_TEMPLATES.items():
        res = cv2.matchTemplate(thresh_corner, suit_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_suit_score and max_val > 0.70:
            best_suit_score = max_val
            best_suit = suit_name

    return f"{best_rank}{best_suit}"

def process_frame(frame):
    """Processes the image to isolate and detect card-like shapes."""
    # Convert to grayscale and blur to remove high-frequency noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image (Binary or Adaptive depending on your lighting)
    # Alternatively, you can use Canny Edge Detection: cv2.Canny(blurred, 30, 150)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cards = 0
    for contour in contours:
        # Filter out tiny artifacts (noise) by enforcing a minimum area
        area = cv2.contourArea(contour)
        if area < 2000:  # Adjust this threshold based on card size
            continue

        # Approximate the contour shape to check if it's rectangular
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Playing cards or ID cards generally have 4 corners
        # Inside your `for contour in contours:` loop, where you find a 4-corner card:
        if len(approx) == 4:
            detected_cards += 1
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

            # 1. Crop the individual card out of the screen frame
            x, y, w, h = cv2.boundingRect(approx)
            card_crop = frame[y:y+h, x:x+w]

            # 2. Extract the top-left corner where rank and suit live
            # (Usually taking the top 20-30% height and left 20-30% width works perfectly)
            corner_w = int(w * 0.25)
            corner_h = int(h * 0.35)
            card_corner = card_crop[0:corner_h, 0:corner_w]

            # 3. Pass this corner to a classification function
            card_text = recognize_card(card_corner)
            print(f"Detected card text: {card_text}")

    # Display count on frame
    cv2.putText(frame, f"Cards: {detected_cards}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def main():
    # Initialize the fast screen grabber
    #with mss.mss() as sct:
    print("Starting screen detection loop. Press 'q' to quit.")

    while True:
        start_time = time.time()

        # Grab raw pixels from the screen ROI
        #screen_img = sct.grab(SCREEN_ROI)
        screen_img = cv2.imread(r"C:\Users\PC\Desktop\pokerpatio.png")
        cropped_roi = screen_img[boardpx["top"]:boardpx["bottom"], boardpx["left"]:boardpx["right"]]

        # Convert raw pixels to a format OpenCV understands (numpy array)
        # mss outputs BGRA, we convert it to BGR
        frame = np.array(cropped_roi)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run the card detection pipeline
        processed_frame = process_frame(frame)

        # Show the visual window
        cv2.imshow("Card Detector", processed_frame)

        # Optional: Print frame rate to console
        print(f"FPS: {1.0 / (time.time() - start_time):.2f}")

        # Break loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
