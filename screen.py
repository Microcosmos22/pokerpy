import cv2
import numpy as np
import time
import os


# =========================================================
# TEMPLATE LOADING
# =========================================================

def load_templates(folder_path):
    """Load all color templates from a folder."""

    templates = {}

    if not os.path.exists(folder_path):
        print(
            f"Error: Folder '{folder_path}' not found!"
        )
        return templates

    for filename in os.listdir(folder_path):

        if not filename.endswith(".png"):
            continue

        name = os.path.splitext(filename)[0]
        path = os.path.join(folder_path, filename)

        img = cv2.imread(
            path,
            cv2.IMREAD_COLOR
        )

        if img is not None:
            templates[name] = img

    print(
        f"Loaded {len(templates)} templates "
        f"from '{folder_path}'"
    )

    return templates


# Load rank and suit templates separately
RANK_TEMPLATES = load_templates(
    "templates/ranks"
)

SUIT_TEMPLATES = load_templates(
    "templates/suits"
)


# =========================================================
# SCREEN REGION
# =========================================================

boardpx = {
    "bottom": 600,
    "left": 600,
    "right": 1150,
    "top": 400
}


# =========================================================
# CORNER SPLIT
# =========================================================
#
# IMPORTANT:
# This must match the split used in extract_templates.py.
#
# For example:
#
# split_ratio = 0.40
#
# means:
#   top 40%    -> rank
#   bottom 60% -> suit
#

split_ratio = 0.40


def match_with_shift(target_img, template_img, max_shift=5):

    best_score = -1
    best_shift = (0, 0)

    h, w = target_img.shape[:2]

    # Template must already have target dimensions
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):

            shifted = np.zeros_like(target_img)

            # Calculate source/destination coordinates
            src_x1 = max(0, -dx)
            src_x2 = min(w, w - dx)

            src_y1 = max(0, -dy)
            src_y2 = min(h, h - dy)

            dst_x1 = max(0, dx)
            dst_x2 = min(w, w + dx)

            dst_y1 = max(0, dy)
            dst_y2 = min(h, h + dy)

            shifted[
                dst_y1:dst_y2,
                dst_x1:dst_x2
            ] = target_img[
                src_y1:src_y2,
                src_x1:src_x2
            ]

            # Compare shifted live image to template
            res = cv2.matchTemplate(
                shifted,
                template_img,
                cv2.TM_CCOEFF_NORMED
            )

            _, score, _, _ = cv2.minMaxLoc(res)

            if score > best_score:
                best_score = score
                best_shift = (dx, dy)

    return best_score, best_shift

# =========================================================
# GENERIC TEMPLATE MATCHING
# =========================================================

def match_templates(
    target_img,
    templates
):
    """
    Compare one live image against a dictionary
    of templates.

    Returns:
        best_match
        highest_score
        all_scores
    """

    if not templates:
        return "??", -1.0, []

    target_h, target_w = target_img.shape[:2]

    all_scores = []

    for name, template_img in templates.items():

        try:
            UPSCALE = 1

            # Resize template to the original live-image dimensions
            resized_template = cv2.resize(template_img,(target_w, target_h),interpolation=cv2.INTER_AREA)

            max_val, best_shift = match_with_shift(target_img,resized_template,max_shift=4)

            all_scores.append((name, max_val))

            print(f"{name}: score={max_val:.4f}, "f"shift={best_shift}")


            overlay = cv2.addWeighted(
                target_img,
                0.5,              # live image weight
                resized_template,
                0.5,              # template weight
                0
            )

            cv2.imwrite(
                "debug_overlay.png",
                overlay
            )

            all_scores.append(
                (name, max_val)
            )

        except Exception as e:

            print(
                f"Matching error for {name}: {e}"
            )

    if not all_scores:
        return "??", -1.0, []

    # Highest score first
    all_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_match, highest_score = all_scores[0]

    return (
        best_match,
        highest_score,
        all_scores
    )


# =========================================================
# RANK RECOGNITION
# =========================================================

def recognize_rank(rank_img):
    """
    Recognize the rank from the rank portion
    of the card corner.
    """

    best_match, highest_score, all_scores = match_templates(
        rank_img,
        RANK_TEMPLATES
    )

    print("\nRank candidates:")

    for rank, score in all_scores[:3]:

        print(
            f"  {rank}: {score:.4f}"
        )

    return best_match, highest_score


# =========================================================
# SUIT RECOGNITION
# =========================================================

def recognize_suit(suit_img):
    """
    Recognize the suit from the suit portion
    of the card corner.
    """

    best_match, highest_score, all_scores = match_templates(
        suit_img,
        SUIT_TEMPLATES
    )

    print("\nSuit candidates:")

    for suit, score in all_scores[:3]:

        print(
            f"  {suit}: {score:.4f}"
        )

    return best_match, highest_score


# =========================================================
# CARD RECOGNITION
# =========================================================

def recognize_card(card_corner):
    """
    Extract rank and suit from the live card corner
    and recognize them independently.
    """

    if not RANK_TEMPLATES or not SUIT_TEMPLATES:
        return "??"

    # -----------------------------------------------------
    # Save live corner for debugging
    # -----------------------------------------------------

    cv2.imwrite("debug_live_corner.png",card_corner)
    corner_h = card_corner.shape[0]

    # -----------------------------------------------------
    # Split rank / suit
    # -----------------------------------------------------

    split_y = int(corner_h * split_ratio) + 5
    rank_img = card_corner[:split_y,:]
    suit_img = card_corner[split_y:,:]

    # Save the individual regions
    cv2.imwrite("debug_live_rank.png",rank_img)
    cv2.imwrite("debug_live_suit.png",suit_img)

    # -----------------------------------------------------
    # Recognize independently
    # -----------------------------------------------------

    rank, rank_score = recognize_rank(rank_img)

    suit, suit_score = recognize_suit(suit_img)

    # -----------------------------------------------------
    # Debug output
    # -----------------------------------------------------

    print(f"\nRecognized rank: {rank} "f"({rank_score:.4f})")

    print(f"Recognized suit: {suit} "f"({suit_score:.4f})")

    # -----------------------------------------------------
    # Confidence thresholds
    # -----------------------------------------------------

    if rank_score <= 0.30:
        print(f"❌ Rank rejected: {rank} "f"({rank_score:.4f})")
        return "??"

    if suit_score <= 0.30:
        print(f"❌ Suit rejected: {suit} "f"({suit_score:.4f})")
        return "??"

    # -----------------------------------------------------
    # Combine rank + suit
    # -----------------------------------------------------

    card_name = f"{rank}{suit}"

    print(f"✅ CARD ACCEPTED: {card_name} "f"(rank={rank_score:.4f}, "f"suit={suit_score:.4f})")
    print("----------------------------")

    return card_name


# =========================================================
# CARD DETECTION
# =========================================================

def process_frame(frame):

    print("\n=== process_frame ===")

    # -----------------------------------------------------
    # CARD DETECTION
    # Grayscale is ONLY used for finding the card shapes.
    # -----------------------------------------------------

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5, 5),0)
    _, thresh = cv2.threshold(blurred,150,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    print(f"Contours found: {len(contours)}")

    detected_cards = 0

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)
        if area < 2000:
            continue

        peri = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,0.04 * peri,True)

        if len(approx) != 4:
            continue

        detected_cards += 1

        x, y, w, h = cv2.boundingRect(approx)

        card_crop = frame[y:y+h,x:x+w]
        corner_w = int(w * 0.33)
        corner_h = int(h * 0.40)

        card_corner = card_crop[0:corner_h,0:corner_w]


        card_text = recognize_card(card_corner)


        cv2.putText(frame,card_text,
            (x,max(y - 10, 20)),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)


    cv2.putText(frame,f"Cards: {detected_cards}",(10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2
    )

    return frame


def main():

    print("Starting screen detection loop. ""Press 'q' to quit.")

    while True:

        start_time = time.time()
        screen_img = cv2.imread(r"C:\Users\PC\Desktop\pokerpatio.png")

        if screen_img is None:
            print("Error: Could not load screenshot.")
            break

        # CROP THE BOARD CARDS
        cropped_roi = screen_img[boardpx["top"]:boardpx["bottom"],boardpx["left"]:boardpx["right"]]

        frame = cropped_roi.copy()

        processed_frame = process_frame(frame)

        cv2.imshow("Card Detector",processed_frame)


        elapsed = (time.time() - start_time)

        if elapsed > 0:

            print(f"FPS: "f"{1.0 / elapsed:.2f}")

        # -------------------------------------------------
        # Quit
        # -------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
