import cv2
import numpy as np

# =====================================================================
# Motion Configuration
# =====================================================================
CONFIG = dict(
    diff_thresh=50, rim_pad=10, speed_gain=1.5, max_speed=40.0, tool_pad=8, min_blob_area=30
)

def plate_circle(box, w, h, pad):
    px1, py1, px2, py2 = map(int, box)
    pw, ph = px2 - px1, py2 - py1
    radius = int(max(pw, ph) / 2) + pad
    cx, cy = px1 + pw // 2, py1 + ph // 2
    edge_margin = 20
    if pw < ph:
        if px1 < edge_margin: cx = px2 - (radius - pad)
        elif px2 > w - edge_margin: cx = px1 + (radius - pad)
    if ph < pw:
        if py1 < edge_margin: cy = py2 - (radius - pad)
        elif py2 > h - edge_margin: cy = py1 + (radius - pad)
    return cx, cy, radius

def detect_motion(prev_bgr, curr_bgr, bin_box, plate_box, prev_plate_box, tool_boxes, prev_tool_boxes):
    h, w = curr_bgr.shape[:2]
    bw, bh = bin_box[2] - bin_box[0], bin_box[3] - bin_box[1]
    
    ix1, iy1 = max(0, int(bin_box[0] + bw * 0.25)), max(0, int(bin_box[1] + bh * 0.25))
    ix2, iy2 = min(w, int(bin_box[2] - bw * 0.25)), min(h, int(bin_box[3] - bh * 0.25))
    if ix2 <= ix1 or iy2 <= iy1: return False, None
    roi_rect = (ix1, iy1, ix2, iy2)

    cur, prv = curr_bgr[iy1:iy2, ix1:ix2], prev_bgr[iy1:iy2, ix1:ix2]

    g_cur = cv2.GaussianBlur(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    g_prv = cv2.GaussianBlur(cv2.cvtColor(prv, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, changed = cv2.threshold(cv2.absdiff(g_cur, g_prv), CONFIG['diff_thresh'], 255, cv2.THRESH_BINARY)

    mask = np.zeros_like(changed)
    circles = [plate_circle(pb, w, h, 0) for pb in (plate_box, prev_plate_box) if pb is not None]
    
    if circles:
        speed = 0.0
        if len(circles) == 2:
            speed = min(float(np.hypot(circles[0][0] - circles[1][0], circles[0][1] - circles[1][1])), CONFIG['max_speed'])
        pad = CONFIG['rim_pad'] + int(CONFIG['speed_gain'] * speed)
        for cx, cy, r in circles:
            cv2.circle(mask, (cx - ix1, cy - iy1), r + pad, 255, -1)
        if len(circles) == 2:
            (ax, ay, ar), (bx, by, br) = circles
            cv2.line(mask, (ax - ix1, ay - iy1), (bx - ix1, by - iy1), 255, 2 * (min(ar, br) + pad))

    for tb in list(tool_boxes) + list(prev_tool_boxes):
        tx1, ty1, tx2, ty2 = map(int, tb)
        p = CONFIG['tool_pad']
        rx1, ry1 = max(0, tx1 - ix1 - p), max(0, ty1 - iy1 - p)
        rx2, ry2 = min(ix2 - ix1, tx2 - ix1 + p), min(iy2 - iy1, ty2 - iy1 + p)
        if rx1 < rx2 and ry1 < ry2: mask[ry1:ry2, rx1:rx2] = 255
            
    changed[mask > 0] = 0
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(changed, connectivity=8)
    
    kept = np.zeros_like(changed)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= CONFIG['min_blob_area']:
            kept[labels == k] = 255

    return cv2.countNonZero(kept) > 0, roi_rect