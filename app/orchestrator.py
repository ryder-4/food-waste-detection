class StateManager:
    def __init__(self):
        self.state = "IDLE"
        self.patience_max = 15
        self.patience = self.patience_max
        self.scraping_frame_count = 0
        self.total_scraping_events = 0
        
        self.motion_streak = 0
        self.required_motion_frames = 3

        self.capture_waste_now = False
        self.snapshot_saved = False
        self.current_event_waste_pct = None
        self.current_event_waste_weight = None
        
        self.candidate_waste_data = None
        self.saved_cx = 0
        self.wait_timer = 0
        self.max_wait_frames = 25

    def _check_overlap(self, boxA, boxB):
        if boxA is None or boxB is None: return False
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        return max(0, xB - xA) * max(0, yB - yA) > 0

    def update(self, plate_box, bin_box, motion_detected, roi_rect, clean_frame, hands, spoons, food_boxes, frame_confs):
        self.capture_waste_now = False
        is_overlapping = self._check_overlap(plate_box, bin_box)
        
        plate_in_roi = False
        if roi_rect is not None and plate_box is not None:
            plate_in_roi = self._check_overlap(plate_box, roi_rect)

        plate_cx = None
        if plate_box is not None:
            plate_cx = (plate_box[0] + plate_box[2]) / 2

        if not self.snapshot_saved:
            if plate_in_roi:
                if self.candidate_waste_data is None:
                    self.saved_cx = plate_cx if plate_cx is not None else 0
                    self.candidate_waste_data = (clean_frame.copy(), plate_box, list(hands), list(spoons), list(food_boxes), list(frame_confs))
                    self.wait_timer = 0
                elif plate_cx is not None and plate_cx - self.saved_cx > 10:
                    self.saved_cx = plate_cx
                    self.candidate_waste_data = (clean_frame.copy(), plate_box, list(hands), list(spoons), list(food_boxes), list(frame_confs))
                    self.wait_timer = 0
                    
            if self.candidate_waste_data is not None:
                self.wait_timer += 1
                if self.wait_timer >= self.max_wait_frames:
                    self.capture_waste_now = True
                    self.snapshot_saved = True

        if self.state == "IDLE":
            if is_overlapping:
                if motion_detected:
                    self.motion_streak += 1
                else:
                    self.motion_streak = 0

                if self.motion_streak >= self.required_motion_frames:
                    self.state = "SCRAPING"
                    self.scraping_frame_count = 1
                    self.patience = self.patience_max
                    self.total_scraping_events += 1
                    self.motion_streak = 0
                    print(f"--- Event {self.total_scraping_events} Started ---")
            else:
                self.motion_streak = 0

        elif self.state == "SCRAPING":
            if is_overlapping:
                self.patience = self.patience_max
                self.scraping_frame_count += 1
            else:
                self.patience -= 1
                if self.patience <= 0:
                    self.state = "IDLE"
                    self.scraping_frame_count = 0
                    self.snapshot_saved = False 
                    self.current_event_waste_pct = None
                    self.current_event_waste_weight = None
                    self.candidate_waste_data = None
                    self.saved_cx = 0
                    self.wait_timer = 0
                    print(f"--- Event {self.total_scraping_events} Ended ---\n")

    def get_current_event_id(self):
        """
        Edge case logic moved from main loop: 
        If the ratcheting snapshot timer finishes BEFORE the scraping event officially 
        starts (State is still IDLE), we need to project the ID + 1 for the files.
        """
        return self.total_scraping_events + (1 if self.state == "IDLE" else 0)

    def set_waste_results(self, waste_pct, wasted_weight):
        """Safely update the orchestrator's state with the calculated metrics."""
        self.current_event_waste_pct = waste_pct
        self.current_event_waste_weight = wasted_weight