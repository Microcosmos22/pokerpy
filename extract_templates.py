import cv2
import numpy as np
import os

def main():
    saved_count = 0
    # 1. Load the master deck sheet
    img_path = "deck_sheet.jpg"  # Rename your downloaded image to this
    deck_img = cv2.imread(img_path)

    if deck_img is None:
        print(f"Error: Could not load {img_path}")
        return

    # Create output directory
    output_dir = "templates"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Preprocess to find card boundaries
    gray = cv2.cvtColor(deck_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV) # Inverse because background is white

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for card-sized contours
    card_contours = []
    card_number=1
    for c in contours:
        area = cv2.contourArea(c)
        if 5000 < area < 50000:  # Adjust based on image resolution
            card_contours.append(c)

            # 1. Get the bounding box coordinates of the contour
            #x, y, w, h = cv2.boundingRect(c)
            #card_crop = deck_img[y:y+h, x:x+w]
            #cv2.imshow(f"Detected Card {card_number}", card_crop)
            #card_number += 1
            #cv2.waitKey(1000)



    print(f"Found {len(card_contours)} potential card shapes. Expecting 60 (including Jokers).")

    if len(card_contours) != 60:
        print("Error: Grid detection mismatched. Ensure your image is cropped cleanly to just the cards.")

    # 3. ROBUST FIXED: Use OpenCV K-Means to cluster Y-centers into 6 exact rows
    # Step 3a: Extract all center Y positions
    centers_y = []
    for c in card_contours:
        x, y, w, h = cv2.boundingRect(c)
        centers_y.append(y + (h // 2))

    # Format data for cv2.kmeans (must be float32)
    centers_y = np.array(centers_y, dtype=np.float32).reshape(-1, 1)

    # Step 3b: Run K-Means to find the 6 true horizontal row coordinates
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(centers_y, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Step 3c: Map the cluster center values back to their actual sorted row index (0 to 5)
    sorted_centers_indices = np.argsort(centers.flatten())
    cluster_to_row_idx = {cluster_id: row_idx for row_idx, cluster_id in enumerate(sorted_centers_indices)}

    # Step 3d: Distribute contours into their mathematically assigned row buckets
    row_buckets = [[] for _ in range(6)]
    for i, c in enumerate(card_contours):
        cluster_id = labels[i][0]
        assigned_row = cluster_to_row_idx[cluster_id]
        row_buckets[assigned_row].append(c)

    # Step 3e: Sort each row bucket strictly left-to-right by X coordinate
    rows = []
    for bucket in row_buckets:
        sorted_row = sorted(bucket, key=lambda c: cv2.boundingRect(c)[0])
        rows.append(sorted_row)

    # 4. Define our grid mapping layout based on the image structure
    # Rows 0-2 (Left 5 = Hearts, Right 5 = Diamonds)
    # Rows 3-5 (Left 5 = Clubs, Right 5 = Spades)
    rank_mapping = {
        0: ['A', '2', '3', '4', '5'],
        1: ['6', '7', '8', '9', '10'],
        2: ['J', 'Q', 'K', 'Joker1', 'Joker2'],
        3: ['A', '2', '3', '4', '5'],
        4: ['6', '7', '8', '9', '10'],
        5: ['J', 'Q', 'K', 'Joker1', 'Joker2']
    }

        # 5. Extract and save the top-left corner templates
    for row_idx, row in enumerate(rows):

        # This index keeps track of our labels (0 to 9) across the row
        # It ONLY increments when we successfully process a good card image!
        name_index = 0

        for col_idx, contour in enumerate(row):
            x, y, w, h = cv2.boundingRect(contour)

            # --- SKIP THE BAD IMAGE HERE ---
            # If this is the Clubs row (Row 3) and it's the bad duplicate image...
            if row_idx == 3 and col_idx == 2:
                print(f"Skipping bad image at column {col_idx}, holding the label for the next card.")
                continue # Discards this bad image instantly. name_index stays the same!
            # -------------------------------

            # Determine Rank using name_index instead of col_idx
            is_left_side = name_index < 5
            lookup_col = name_index if is_left_side else name_index - 5
            rank = rank_mapping[row_idx][lookup_col]

            # --- FIX FOR MISSING FACE CARDS ---
            # We increment the name index for the layout column BEFORE skipping the Joker.
            # This ensures that columns 5, 6, and 7 (J, Q, K) line up correctly on the right side.
            if "Joker" in rank:
                name_index += 1
                continue
            # ----------------------------------

            # 1. Process the valid card crop
            card_crop = deck_img[y:y+h, x:x+w]
            corner_w = int(w * 0.25)
            corner_h = int(h * 0.35)
            corner_crop = card_crop[0:corner_h, 0:corner_w]

            # 3. Determine Suit
            if row_idx < 3:
                suit = 'h' if is_left_side else 'd' # Hearts / Diamonds
            else:
                suit = 'c' if is_left_side else 's' # Clubs / Spades

            card_name = f"{rank}{suit}"

            filename = f"{card_name}.png"
            filepath = os.path.join(output_dir, filename)

            print(f"Writing file: {filename} from Row {row_idx}, Col {col_idx}")
            cv2.imwrite(filepath, corner_crop)
            saved_count += 1

            # 5. Advance our naming tracker now that this card is safely archived
            name_index += 1

    print(f"\n Done! Successfully generated {saved_count} card templates inside your '/{output_dir}' directory!")


if __name__ == "__main__":
    main()
