
import cv2
import numpy as np
import os


def main():
    saved_count = 0

    # 1. Load the master deck sheet
    img_path = "deck_sheet.jpg"
    deck_img = cv2.imread(img_path)

    if deck_img is None:
        print(f"Error: Could not load {img_path}")
        return

    # Create output directory
    output_dir = "templates"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Preprocess to find card boundaries
    # Use grayscale + threshold ONLY for contour detection.
    # The original color image (deck_img) is kept for template extraction.
    gray = cv2.cvtColor(deck_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        245,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter for card-sized contours
    card_contours = []

    for c in contours:
        area = cv2.contourArea(c)

        if 5000 < area < 50000:
            card_contours.append(c)

    print(
        f"Found {len(card_contours)} potential card shapes. "
        f"Expecting 60 (including Jokers)."
    )

    if len(card_contours) != 60:
        print(
            "Error: Grid detection mismatched. "
            "Ensure your image is cropped cleanly to just the cards."
        )

    # 3. Use OpenCV K-Means to cluster Y-centers into 6 exact rows

    # Step 3a: Extract all center Y positions
    centers_y = []

    for c in card_contours:
        x, y, w, h = cv2.boundingRect(c)
        centers_y.append(y + (h // 2))

    # Format data for cv2.kmeans (must be float32)
    centers_y = np.array(
        centers_y,
        dtype=np.float32
    ).reshape(-1, 1)

    # Step 3b: Run K-Means to find the 6 true horizontal row coordinates
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        centers_y,
        6,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Step 3c: Map cluster center values back to sorted row index
    sorted_centers_indices = np.argsort(centers.flatten())

    cluster_to_row_idx = {
        cluster_id: row_idx
        for row_idx, cluster_id in enumerate(sorted_centers_indices)
    }

    # Step 3d: Distribute contours into row buckets
    row_buckets = [[] for _ in range(6)]

    for i, c in enumerate(card_contours):
        cluster_id = labels[i][0]
        assigned_row = cluster_to_row_idx[cluster_id]
        row_buckets[assigned_row].append(c)

    # Step 3e: Sort each row strictly left-to-right
    rows = []

    for bucket in row_buckets:
        sorted_row = sorted(
            bucket,
            key=lambda c: cv2.boundingRect(c)[0]
        )
        rows.append(sorted_row)

    # 4. Define grid mapping layout
    rank_mapping = {
        0: ['A', '2', '3', '4', '5'],
        1: ['6', '7', '8', '9', '10'],
        2: ['J', 'Q', 'K', 'Joker1', 'Joker2'],
        3: ['A', '2', '3', '4', '5'],
        4: ['6', '7', '8', '9', '10'],
        5: ['J', 'Q', 'K', 'Joker1', 'Joker2']
    }

    # 5. Extract and save the top-left corner templates
    # The crop comes directly from deck_img, so it remains COLOR.
    for row_idx, row in enumerate(rows):

        # Tracks our labels across the row
        name_index = 0

        for col_idx, contour in enumerate(row):

            x, y, w, h = cv2.boundingRect(contour)

            # Skip the known bad image
            if row_idx == 3 and col_idx == 2:
                print(
                    f"Skipping bad image at column {col_idx}, "
                    f"holding the label for the next card."
                )
                continue

            # Determine rank using name_index
            is_left_side = name_index < 5
            lookup_col = (
                name_index
                if is_left_side
                else name_index - 5
            )

            rank = rank_mapping[row_idx][lookup_col]

            # Skip Jokers
            if "Joker" in rank:
                name_index += 1
                continue

            # Process the valid card crop.
            # IMPORTANT: card_crop comes from the ORIGINAL COLOR image.
            card_crop = deck_img[y:y+h, x:x+w]

            corner_w = int(w * 0.17)
            corner_h = int(h * 0.25)

            corner_crop = card_crop[
                4:corner_h,
                3:corner_w
            ]

            # 6. Determine Suit
            if row_idx < 3:
                suit = 'h' if is_left_side else 'd'
            else:
                suit = 'c' if is_left_side else 's'

            card_name = f"{rank}{suit}"

            filename = f"{card_name}.png"
            filepath = os.path.join(
                output_dir,
                filename
            )

            print(
                f"Writing file: {filename} "
                f"from Row {row_idx}, Col {col_idx}"
            )

            # Save the COLOR template
            cv2.imwrite(
                filepath,
                corner_crop
            )

            saved_count += 1

            # Advance naming tracker
            name_index += 1

    print(
        f"\nDone! Successfully generated "
        f"{saved_count} color card templates "
        f"inside your '/{output_dir}' directory!"
    )


if __name__ == "__main__":
    main()
