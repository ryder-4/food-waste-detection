import cv2
import os
import csv
import glob
import sys
from ultralytics import YOLO

# Local Module Imports
from orchestrator import StateManager
from waste_event_detection import detect_motion, plate_circle
from amount_detection import calculate_waste_from_frame
from weight_detection import classify_meal, calculate_wasted_weight

CLASS_NAMES = [
    'bin', 'biryani', 'burger', 'carrot', 'chicken', 'cucumber', 
    'drumstick', 'fries', 'hand', 'onion', 'pepper-chilli', 
    'plate', 'sandwich', 'spoon'
]

def run_pipeline(video_path, model_weights, out_dir='../output'):
    os.makedirs(out_dir, exist_ok=True)
    out_video_path = os.path.join(out_dir, 'pipeline_output.mp4')

    model = YOLO(model_weights)
    manager = StateManager()
    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    prev_clean = None
    prev_plate_box = None
    prev_tool_boxes = []

    # --- CSV Tracking Variables ---
    events_log = {}
    prev_state = "IDLE"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame % 30 == 0:
            print(f"Processing frame {current_frame}...")

        clean = frame.copy()
        results = model.predict(frame, conf=0.8, verbose=False)

        plate_box = None
        bin_box = None
        tool_boxes = []
        hands = []
        spoons = []
        food_boxes = []
        frame_confs = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            
            frame_confs.append(conf)

            if cls_id == 0 and bin_box is None:  
                bin_box = coords
                cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (255, 0, 0), 2)
                cv2.putText(frame, "bin", (int(coords[0]), int(coords[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif cls_id == 11 and plate_box is None:  
                plate_box = coords
                cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 255), 2)
                cv2.putText(frame, "plate", (int(coords[0]), int(coords[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif cls_id == 8:  
                hands.append(coords)
                tool_boxes.append(coords)
            elif cls_id == 13:  
                spoons.append(coords)
                tool_boxes.append(coords)
            else:
                food_boxes.append((cls_id, coords))
                cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
                cv2.putText(frame, CLASS_NAMES[cls_id], (int(coords[0]), int(coords[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw and label hands and spoons
        for tb in hands:
            cv2.rectangle(frame, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), (255, 165, 0), 2)
            cv2.putText(frame, "hand", (int(tb[0]), int(tb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        for tb in spoons:
            cv2.rectangle(frame, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), (255, 0, 255), 2)
            cv2.putText(frame, "spoon", (int(tb[0]), int(tb[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        motion_detected = False
        roi = None
        if bin_box is not None and prev_clean is not None:
            motion_detected, roi = detect_motion(
                prev_clean, clean, bin_box, plate_box, prev_plate_box, tool_boxes, prev_tool_boxes
            )

            if roi is not None:
                ix1, iy1, ix2, iy2 = roi
                roi_color = (0, 0, 255) if motion_detected else (255, 255, 255)
                cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), roi_color, 2)
                cv2.putText(frame, f"ROI (Streak: {manager.motion_streak})", (ix1, iy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2)

        if plate_box is not None: 
            cx, cy, r = plate_circle(plate_box, w, h, 8)
            cv2.circle(frame, (cx, cy), r, (0, 0, 0), 2)

        prev_clean = clean
        prev_plate_box = plate_box
        prev_tool_boxes = tool_boxes

        manager.update(plate_box, bin_box, motion_detected, roi, clean, hands, spoons, food_boxes, frame_confs)

        # --- CSV: Catch Event Start Boundary ---
        if manager.state == "SCRAPING" and prev_state == "IDLE":
            event_id = manager.total_scraping_events
            if event_id not in events_log:
                events_log[event_id] = {}
            events_log[event_id]['start_frame'] = current_frame
            events_log[event_id]['start_timestamp'] = current_frame / fps

        # -----------------------------------------------------------
        # Snapshot Triggered: Process Images, Weight, and Classification
        # -----------------------------------------------------------
        if manager.capture_waste_now and manager.candidate_waste_data is not None:
            c_frame, c_plate, c_hands, c_spoons, c_foods, c_confs = manager.candidate_waste_data
            
            # --- 1. Classify Meal (Delegated to weight_detection.py) ---
            detected_food_names = [CLASS_NAMES[f_cls] for f_cls, _ in c_foods]
            meal_name, shrink_pad, total_meal_weight = classify_meal(detected_food_names)

            # --- 2. Calculate Amount Wasted (Delegated to amount_detection.py) ---
            waste_pct, amount_img = calculate_waste_from_frame(c_frame, c_plate, c_hands, c_spoons, shrink_pad)
            
            # --- 3. Calculate Weight Wasted (Delegated to weight_detection.py) ---
            wasted_weight = calculate_wasted_weight(total_meal_weight, waste_pct)
            
            avg_conf = sum(c_confs) / len(c_confs) if c_confs else 0.0
            
            manager.set_waste_results(waste_pct, wasted_weight)
            
            # --- CSV: Record Calculations ---
            event_id = manager.get_current_event_id()
            if event_id not in events_log:
                events_log[event_id] = {}
            events_log[event_id].update({
                'meal': meal_name,
                'amount': waste_pct,
                'weight': wasted_weight,
                'confidence': avg_conf
            })
            
            # --- 4. Generate Output Images ---
            clean_img_name = os.path.join(out_dir, f"event_{event_id}_clean_frame.jpg")
            cv2.imwrite(clean_img_name, c_frame)
            
            amount_img_name = os.path.join(out_dir, f"event_{event_id}_amount_wasted.jpg")
            cv2.imwrite(amount_img_name, amount_img)
            
            weight_img = c_frame.copy()
            for f_cls, f_box in c_foods:
                bx1, by1, bx2, by2 = map(int, f_box)
                cv2.rectangle(weight_img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(weight_img, CLASS_NAMES[f_cls], (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(weight_img, f"weight wasted = {wasted_weight:.1f} grams", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            text_size = cv2.getTextSize(meal_name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.putText(weight_img, meal_name, (w - text_size[0] - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            weight_img_name = os.path.join(out_dir, f"event_{event_id}_weight_wasted.jpg")
            cv2.imwrite(weight_img_name, weight_img)
            
            print(f"    [!] Meal Classified: {meal_name} (Inner Pad: {shrink_pad}px, Total Meal Weight: {total_meal_weight}g)")
            print(f"    [!] Waste calculated: {waste_pct:.1f}% -> {wasted_weight:.1f}g wasted")
            print(f"        -> Saved to {out_dir}")

        # --- CSV: Catch Event End Boundary ---
        if manager.state == "IDLE" and prev_state == "SCRAPING":
            event_id = manager.total_scraping_events
            if event_id not in events_log:
                events_log[event_id] = {}
            events_log[event_id]['end_frame'] = current_frame
            events_log[event_id]['end_timestamp'] = current_frame / fps

        prev_state = manager.state

        # Overlay text on video
        status_color = (0, 255, 0) if manager.state == "SCRAPING" else (0, 0, 255)
        cv2.putText(frame, f"STATE: {manager.state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        cv2.putText(frame, f"Patience: {manager.patience}/{manager.patience_max}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Updated text overlay per request
        if manager.current_event_waste_pct is not None and manager.current_event_waste_weight is not None:
            text_pct = f"%age of full meal = {manager.current_event_waste_pct:.1f}"
            size_pct = cv2.getTextSize(text_pct, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
            cv2.putText(frame, text_pct, (w - size_pct[0] - 30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
            
            text_wgt = f"total weight = {manager.current_event_waste_weight:.1f} grams"
            size_wgt = cv2.getTextSize(text_wgt, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
            cv2.putText(frame, text_wgt, (w - size_wgt[0] - 30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        out.write(frame)

    cap.release()
    out.release()
    
    # -----------------------------------------------------------
    # Finalize CSV File Logging
    # -----------------------------------------------------------
    csv_path = os.path.join(out_dir, 'wastage_log.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Event_ID', 'Start_Timestamp(s)', 'End_Timestamp(s)', 'Classified_Meal', 'Start_Frame', 'End_Frame', 'Amount_Wasted(%)', 'Weight_Wasted(g)', 'Confidence_Score'])
        
        for eid, data in events_log.items():
            writer.writerow([
                eid,
                round(data.get('start_timestamp', 0.0), 2),
                round(data.get('end_timestamp', 0.0), 2),
                data.get('meal', 'Unknown'),
                data.get('start_frame', 0),
                data.get('end_frame', 0),
                round(data.get('amount', 0.0), 2),
                round(data.get('weight', 0.0), 2),
                round(data.get('confidence', 0.0), 2)
            ])
            
    print(f"Pipeline execution finished. Saved video to {out_video_path}")
    print(f"Saved CSV log to {csv_path}")

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Food Waste Detection Pipeline")
    parser.add_argument('--video', type=str, default=None, help='Path to input video')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='Path to YOLO weights')
    parser.add_argument('--out_dir', type=str, default='../output', help='Directory for output files')
    args = parser.parse_args()

    video_path = args.video

    if not video_path:
        supported_formats = ('*.MOV', '*.mp4', '*.avi', '*.mkv', '*.mov', '*.MP4')
        video_files = []
        for fmt in supported_formats:
            video_files.extend(glob.glob(os.path.join('../data', fmt)))
        
        if not video_files:
            print("Error: No video file found in the 'data' folder. Please place your test video there.")
            sys.exit(1)
        
        video_path = video_files[0]
        print(f"[+] Auto-detected video file: {video_path}")

    run_pipeline(
        video_path=video_path,
        model_weights=args.weights,
        out_dir=args.out_dir
    )