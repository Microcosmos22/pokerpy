
import cv2
import numpy as np
import time
import os


def load_combined_templates(folder_path="templates"):
    """Load all saved color card templates into memory."""
    templates = {}

    if not os.path.exists(folder_path):
        print(
            f"Error: Folder '{folder_path}' not found! "
            "Run your extraction script first."
        )
        return templates

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # Strip extension to get card label, e.g. "3c", "Ah"
            card_name = os.path.splitext(filename)[0]
            path = os.path.join(folder_path, filename)

            # Load template in COLOR
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is not None:
                templates[card_name] = img

    print(f"Loaded {len(templates)} color card templates into memory.")
    return templates


# Initialize templates at script startup
ALL_CARDS_TEMPLATES = load_combined_templates("templates")


# Region of interest on the screen
boardpx = {
    "bottom": 600,
    "left": 600,
    "right": 1150,
    "top": 400
}


def recognize_card(card_corner):
    """
    Compare the incoming live card corner against
    the color templates.
    """

    if not ALL_CARDS_TEMPLATES:
        return "??"

    # ---------------------------------------------------------
    # IMPORTANT:
    # Keep the live corner in COLOR.
    # No grayscale conversion or thresholding here.
    # ---------------------------------------------------------

    target_h, target_w = card_corner.shape[:2]

    all_scores = []

    # Compare against every template
    for card_name, template_img in ALL_CARDS_TEMPLATES.items():

        try:
            # Resize template to exactly match live corner dimensions
            resized_template = cv2.resize(
                template_img,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA
            )

            # -------------------------------------------------
            # COLOR TEMPLATE MATCHING
            #
            # Both images are BGR, 3-channel images.
            # OpenCV's matchTemplate can compare all channels.
            # -------------------------------------------------

            res = cv2.matchTemplate(
                card_corner,
                resized_template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, _ = cv2.minMaxLoc(res)

            all_scores.append((card_name, max_val))

        except Exception as e:
            print(f"Matching error for {card_name}: {e}")
            continue

    if not all_scores:
        return "??"

    # Sort scores from highest to lowest
    all_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_match, highest_score = all_scores[0]

    # ---------------------------------------------------------
    # Debug output
    # ---------------------------------------------------------

    # Save the actual COLOR live corner
    cv2.imwrite(
        "debug_live_corner.png",
        card_corner
    )

    # Save the best matching COLOR template
    best_template = ALL_CARDS_TEMPLATES[best_match]

    best_template_resized = cv2.resize(
        best_template,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )

    cv2.imwrite(
        "debug_live_template.png",
        best_template_resized
    )

    # Print top 3 candidates
    print("\nTop 3 Candidates Evaluated:")

    for rank, (card_name, score) in enumerate(
        all_scores[:3],
        start=1
    ):
        print(
            f"  {rank}. Mapped: {card_name} | "
            f"Score: {score:.4f}"
        )

    # Confidence threshold
    if highest_score <= 0.30:
        print(
            f"❌ REJECTED: {best_match} "
            f"({highest_score:.4f})"
        )
        print("----------------------------")
        return "??"

    print(
        f"✅ ACCEPTED: {best_match} "
        f"({highest_score:.4f})"
    )
    print("----------------------------")

    return best_match


def process_frame(frame):
    print("\n=== process_frame ===")

    # ---------------------------------------------------------
    # CARD DETECTION
    #
    # We still use grayscale here because we only need to
    # identify the card contours.
    # ---------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blurred,
        150,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Contours found: {len(contours)}")

    detected_cards = 0

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        if area < 2000:
            continue

        peri = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.04 * peri,
            True
        )

        if len(approx) == 4:

            detected_cards += 1

            x, y, w, h = cv2.boundingRect(
                approx
            )

            # -------------------------------------------------
            # IMPORTANT:
            # card_crop comes directly from the original
            # COLOR frame.
            # -------------------------------------------------

            card_crop = frame[
                y:y+h,
                x:x+w
            ]

            corner_w = int(w * 0.3)
            corner_h = int(h * 0.42)

            # COLOR corner
            card_corner = card_crop[
                0:corner_h,
                0:corner_w
            ]

            # Recognize using COLOR template matching
            card_text = recognize_card(
                card_corner
            )

            # Display recognized card above the detected card
            cv2.putText(
                frame,
                card_text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    print(
        f"Total cards detected: {detected_cards}"
    )

    cv2.putText(
        frame,
        f"Cards: {detected_cards}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return frame


def main():

    print(
        "Starting screen detection loop. "
        "Press 'q' to quit."
    )

    while True:

        start_time = time.time()

        # Load screenshot
        screen_img = cv2.imread(
            r"C:\Users\PC\Desktop\pokerpatio.png"
        )

        if screen_img is None:
            print("Error: Could not load screenshot.")
            break

        # Crop the board ROI
        cropped_roi = screen_img[
            boardpx["top"]:boardpx["bottom"],
            boardpx["left"]:boardpx["right"]
        ]

        # imread() already returns BGR.
        # No BGRA -> BGR conversion is necessary.
        frame = cropped_roi.copy()

        # Run card detection + recognition
        processed_frame = process_frame(
            frame
        )

        # Show result
        cv2.imshow(
            "Card Detector",
            processed_frame
        )

        # FPS
        elapsed = time.time() - start_time

        if elapsed > 0:
            print(
                f"FPS: {1.0 / elapsed:.2f}"
            )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
