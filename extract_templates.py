
import cv2
import numpy as np
import os


def main():
    saved_ranks = set()
    saved_suits = set()

    # 1. Load the master deck sheet
    img_path = "deck_sheet.jpg"
    deck_img = cv2.imread(img_path)

    if deck_img is None:
        print(f"Error: Could not load {img_path}")
        return

    # Create output directories
    output_dir = "templates"
    ranks_dir = os.path.join(output_dir, "ranks")
    suits_dir = os.path.join(output_dir, "suits")

    os.makedirs(ranks_dir, exist_ok=True)
    os.makedirs(suits_dir, exist_ok=True)

    # 2. Preprocess to find card boundaries
    # Grayscale + threshold are ONLY used for contour detection.
    # The original color image is kept for extracting the templates.
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

    centers_y = []

    for c in card_contours:
        x, y, w, h = cv2.boundingRect(c)
        centers_y.append(y + (h // 2))

    centers_y = np.array(
        centers_y,
        dtype=np.float32
    ).reshape(-1, 1)

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

    # Map cluster centers to sorted row indices
    sorted_centers_indices = np.argsort(
        centers.flatten()
    )

    cluster_to_row_idx = {
        cluster_id: row_idx
        for row_idx, cluster_id
        in enumerate(sorted_centers_indices)
    }

    # Distribute contours into row buckets
    row_buckets = [[] for _ in range(6)]

    for i, c in enumerate(card_contours):
        cluster_id = labels[i][0]
        assigned_row = cluster_to_row_idx[cluster_id]
        row_buckets[assigned_row].append(c)

    # Sort each row strictly left-to-right
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

    # 5. Extract rank and suit templates
    for row_idx, row in enumerate(rows):

        # Tracks labels across the row
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

            # Determine whether we are on the left or right side
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

            # Determine suit from the grid
            if row_idx < 3:
                suit = 'h' if is_left_side else 'd'
            else:
                suit = 'c' if is_left_side else 's'

            # -------------------------------------------------
            # Extract the card from the ORIGINAL COLOR image
            # -------------------------------------------------

            card_crop = deck_img[
                y:y+h,
                x:x+w
            ]

            # -------------------------------------------------
            # Extract the top-left corner
            # -------------------------------------------------

            corner_w = int(w * 0.16)
            corner_h = int(h * 0.23)

            corner_crop = card_crop[22:corner_h,13:corner_w]

            # -------------------------------------------------
            # Split corner into rank and suit
            #
            # Top half    -> rank
            # Bottom half -> suit
            # -------------------------------------------------

            split_y = corner_crop.shape[0] // 2 + 8

            rank_crop = corner_crop[:split_y,:]

            suit_crop = corner_crop[split_y:,:]

            # -------------------------------------------------
            # Save rank template
            #
            # Only save each rank once.
            # -------------------------------------------------

            if rank not in saved_ranks:

                rank_filename = f"{rank}.png"
                rank_filepath = os.path.join(ranks_dir,rank_filename)

                cv2.imwrite(rank_filepath,rank_crop)

                saved_ranks.add(rank)

                print(f"Saved rank template: {rank_filename} "f"from Row {row_idx}, Col {col_idx}")

            # -------------------------------------------------
            # Save suit template
            #
            # Only save each suit once.
            # -------------------------------------------------

            if suit not in saved_suits:

                suit_filename = f"{suit}.png"
                suit_filepath = os.path.join(suits_dir,suit_filename)

                cv2.imwrite(suit_filepath,suit_crop)

                saved_suits.add(suit)

                print(f"Saved suit template: {suit_filename} "f"from Row {row_idx}, Col {col_idx}")

            # Advance naming tracker
            name_index += 1

    print("\nDone!")
    print(
        f"Saved {len(saved_ranks)} rank templates "
        f"to '{ranks_dir}'"
    )
    print(
        f"Saved {len(saved_suits)} suit templates "
        f"to '{suits_dir}'"
    )


if __name__ == "__main__":
    main()
