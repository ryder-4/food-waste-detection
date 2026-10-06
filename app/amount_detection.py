import cv2
import numpy as np

def plate_ellipse(box, w, h, shrink_pad):
    px1, py1, px2, py2 = map(int, box)
    pw, ph = px2 - px1, py2 - py1
    cx, cy = px1 + pw // 2, py1 + ph // 2
    axis_x, axis_y = pw // 2, ph // 2
    edge_margin = 20
    
    if pw < ph * 0.8:
        axis_x = ph // 2  
        if px1 < edge_margin: cx = px2 - axis_x
        elif px2 > w - edge_margin: cx = px1 + axis_x
    elif ph < pw * 0.8:
        axis_y = pw // 2  
        if py1 < edge_margin: cy = py2 - axis_y
        elif py2 > h - edge_margin: cy = py1 + axis_y
            
    axis_x = max(1, axis_x - shrink_pad)
    axis_y = max(1, axis_y - shrink_pad)
    return cx, cy, axis_x, axis_y

def calculate_waste_from_frame(img, plate_box, hands, spoons, shrink_pad):
    h, w = img.shape[:2]
    if plate_box is None:
        return 0.0, img.copy()

    plate_cy = (plate_box[1] + plate_box[3]) / 2
    top_hand = None
    tools = []
    
    for h_box in hands:
        h_box_cy = (h_box[1] + h_box[3]) / 2
        if top_hand is None and h_box_cy < plate_cy:
            top_hand = h_box
        else:
            tools.append(h_box)
    tools.extend(spoons)

    # 1. Inner plate mask (shrunk dynamically)
    plate_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy, axis_x, axis_y = plate_ellipse(plate_box, w, h, shrink_pad=shrink_pad)
    cv2.ellipse(plate_mask, (cx, cy), (axis_x, axis_y), 0, 0, 360, 255, -1)

    # 2. Hand/Tool occlusion masks
    top_hand_mask = np.zeros((h, w), dtype=np.uint8)
    if top_hand:
        tx1, ty1, tx2, ty2 = map(int, top_hand)
        cv2.rectangle(top_hand_mask, (tx1, ty1), (tx2, ty2), 255, -1)

    tools_mask = np.zeros((h, w), dtype=np.uint8)
    for tb in tools:
        tx1, ty1, tx2, ty2 = map(int, tb)
        pad = 5
        cv2.rectangle(tools_mask, (max(0, tx1-pad), max(0, ty1-pad)), 
                                  (min(w, tx2+pad), min(h, ty2+pad)), 255, -1)
        
    occupied_mask = cv2.bitwise_or(top_hand_mask, tools_mask)
    visible_plate_mask = cv2.bitwise_and(plate_mask, cv2.bitwise_not(occupied_mask))

    # 3. HSV Empty Plate Detection
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 60])
    upper_white = np.array([180, 35, 255])
    white_color_mask = cv2.inRange(hsv_img, lower_white, upper_white)

    # 4. Calculation (Purely within the inner circle)
    total_visible_pixels = cv2.countNonZero(visible_plate_mask)
    visible_white = cv2.countNonZero(cv2.bitwise_and(visible_plate_mask, white_color_mask))
    visible_food = total_visible_pixels - visible_white
    
    if total_visible_pixels == 0: 
        wasted_percentage = 0.0
    else:
        wasted_percentage = (visible_food / total_visible_pixels) * 100.0

    # 5. Generate Amount Wasted Debug Image
    debug_img = img.copy()
    debug_img[cv2.bitwise_and(visible_plate_mask, white_color_mask) == 255] = [255, 0, 0] 
    debug_img[cv2.bitwise_and(visible_plate_mask, cv2.bitwise_not(white_color_mask)) == 255] = [0, 255, 0] 
    
    cv2.putText(debug_img, f"%age of the full single serving wasted = {wasted_percentage:.1f}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return wasted_percentage, debug_img