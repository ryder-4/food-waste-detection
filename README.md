# Food Waste Detection

## Introduction
This README explains my approach to solving the problem. After the architecture section, each of the 3 deliverables (waste detection events, amount wasted, and weight wasted) is covered in 3 subsections:
1. My approach and edge cases catered.
2. How I see this solution running in production.
3. What I could have done better.

At the end, there is a guide on [How to run the project](#how-to-run-the-project).

---

## Architecture

The pipeline runs each frame of the input video through an object detection model, then splits the logic into separate detection and estimation modules.

```text
Input Video ---> Object Detection (YOLO) ---> Application
                                                 |
        +-----------------------+----------------+-----------------------+
        |                       |                                        |
        v                       v                                        v
Waste Event Detection   Waste Amount Detection                  Waste Weight Detection
```

| File | Responsibility |
|------|----------------|
| `app/main.py` | Runs the pipeline frame by frame, draws the overlays and writes the video, images and CSV |
| `app/orchestrator.py` | State manager (`IDLE` / `SCRAPING`) and snapshot timing |
| `app/waste_event_detection.py` | Motion detection inside the bin |
| `app/amount_detection.py` | Percentage of the plate covered by food |
| `app/weight_detection.py` + `app/meal_weights.json` | Meal classification and weight in grams |

### Object Detection Model

I used a **YOLO11-L** model (Ultralytics). I trained it on frames taken from the videos provided and annotated them in **Roboflow**.

This was not a food or object detection task, so my goal was not to detect every relevant object perfectly in every frame. Instead, I picked the frames that would actually help produce the deliverables (for example, frames where the plate is over the bin) and annotated those. Choosing frames this way saved both annotation time and training compute.

| Item | Value |
|------|-------|
| Classes | bin, plate, hand, spoon, biryani, burger, carrot, chicken, cucumber, drumstick, fries, onion, pepper-chilli, sandwich |
| Frames annotated | 417 |
| After augmentation | 1001 |
| Train / validation split | 876 training / 125 validation |
| Augmentations | Rotation, hue, saturation |
| Epochs trained | 43 |
| mAP@0.5 | 0.51 |

The pipeline only uses detections with a confidence of **0.8 or higher**.

> **Note on the output video:** You may see inconsistent labelling or classification of food items and spoons in the output video. This is expected, because the annotations were chosen to serve the deliverables rather than general detection. For example, a spoon may not be detected when it is away from the ROI. The model was never trained on such frames, because the pipeline does not need to detect a spoon away from the ROI.

---

## Waste Event Detection

### 1. My Approach and Edge Cases Catered

My approach uses a state manager to track whether the system is **"IDLE"** or actively **"SCRAPING"**. To decide when an event starts and stops, the system goes through these steps:

- The **YOLO model** finds the key objects in each frame: the bin, plate, hands, and spoons.
- It checks whether the plate overlaps the bin.
- It looks for physical motion in a smaller **Region of Interest (ROI)**: the central half of the bin, where the food lands.
- To trigger the **"SCRAPING"** state, motion must be detected while the plate overlaps the bin for **3 consecutive frames**. This stops one noisy frame from starting an event.
- The system also takes a **snapshot of the clean frame** (with no annotations drawn on it) to lock in how much food is on the plate before it is thrown away. The snapshot is taken when the plate moves over the ROI. It is refreshed each time the plate slides further over the bin, and finalised once the plate has stayed still for 25 frames.

#### Edge Cases

- **Plate and tool movement:** The motion detector ignores normal movement of the plate and of tools (hands/spoons) by hiding them with masks. The faster the plate moves, the larger the mask around it, so its motion blur is not counted as falling food.
- **Noise:** Tiny changes (blobs smaller than 30 pixels) are ignored.
- **Patience timer:** If the plate and bin stop overlapping for a split second, the system waits **15 frames** before ending the event. This prevents one scraping action from being split into multiple events.
- **Snapshot before the event starts:** Sometimes the snapshot is finalised just before the `SCRAPING` state begins. In that case, the results are assigned to the upcoming event ID, so they are not lost or attached to the previous event.

### 2. How I See This Solution Running in Production

Some automation tasks give the best results with a little amount of human help. Here, that would mean restaurant employees throwing away all the food once the plate has entered the ROI, before it leaves it. This keeps each event clean and makes edge cases less frequent.

### 3. What I Could Have Done Better

- **More varied data:** I could have collected more data with food being thrown away from more directions. The snapshot logic currently assumes the plate approaches the bin from one side, matching the camera setup in the provided footage.
- **Multiple plates:** The system tracks one plate at a time. I could have handled several plates being emptied into the bin at once.

---

## Waste Amount Detection

### 1. My Approach and Edge Cases Catered

My approach works out how much food is on the plate by looking only at the inner part of the plate and finding the empty white space.

#### Inner Circle Intuition

If we used the whole plate bounding box, the camera would see the empty rim of the plate and count it as empty space. This should not be the case as the rim would be empty is case of a full meal on the plate as well.

To fix this, I draw a smaller **inner ellipse** that cuts out the rim using a **"shrink pad"**.

#### Masking Strategy

I draw boxes over any detected hands or spoons to block them out, so a hand covering the plate is not counted as food.

The system then looks for white pixels (empty plate) inside the inner ellipse, and counts everything that is not white as food:

```text
Amount wasted (%) = food pixels / visible pixels inside the inner ellipse × 100
```

#### Differences for Meals

Different meals need different inner circles because of how the food sits on the plate.

- For a **"KFC Meal"** or **"Sandwich Meal"**, the ellipse is shrunk inward by **50 pixels** for a tighter look at the food.
- For a **"Biryani Meal"** or **"Salad Meal"**, it is shrunk by only **25 pixels**.

The main edge cases catered to are:

- Hands and spoons blocking the food.
- The plate rim, including when the plate is partly out of frame, ruining the calculation.

These are handled with the hand-masking and inner-ellipse padding techniques.

This part of the assignment reminded me of my Computer Vision professor, who told us in class to only use a model when necessary and never to underestimate the power of classical computer vision. Without colour masking, I would have needed a segmentation model to find the food pixels.

### 2. How I See This Solution Running in Production

In production, this step would run the moment a snapshot is locked in for an event.

Kitchen staff would not need to weigh plates by hand. The application would:

1. Take the snapshot image.
2. Calculate the percentage of leftover food.
3. Save a highlighted image showing exactly what it counted as food (green) versus empty plate (blue).

This creates a visual record the restaurant can easily check.

However, this strategy is not very accurate on its own, because a top-down view cannot see depth:

- Food can be **stacked up** and cover less of the plate's surface, so the pipeline would **underestimate** the waste.
- A small amount of rice could be **spread out** across the plate and look like a full meal, so the pipeline would **overestimate** the waste.

A production version would need **multiple camera angles** on the plate to estimate the depth of the food.

### 3. What I Could Have Done Better

The current method assumes the empty parts of the plate are always **white**.

If the restaurant used dark blue or patterned plates, this colour check would fail. Pale foods (such as plain rice) can also be mistaken for empty plate.

I could have improved this with a model that learns the difference between **"food"** and **"plate background"** regardless of the plate's colour.

---

## Waste Weight Detection

### 1. My Approach and Edge Cases Catered

My approach converts the visual percentage of wasted food into a weight in grams using predefined values.

The food weights were gathered through secondary research, cited in [References](#references). For example, one Instagram creator weighed the entire KFC menu, and KFC was one of the meals in the footage provided.

The process works as follows:

1. YOLO detects the food items on the plate.
2. The detected items are used to classify the meal type, checked in this order:
   - Burger or drumstick → **KFC Meal**
   - Sandwich → **Sandwich Meal**
   - At least 2 salad items → **Salad Meal**
   - Biryani or chicken → **Biryani Meal**
3. A baseline weight for the meal is calculated by adding up its components from `meal_weights.json`.
4. The **"Salad Meal"** uses a specific recipe:
   - 1 carrot
   - 1 cucumber
   - 0.25 of an onion
   - 2 peppers/chillies
5. Finally, the total meal weight is multiplied by the waste percentage to get the wasted weight.

| Meal | Components | Baseline weight |
|------|------------|-----------------|
| KFC Meal | Burger (230g) + drumstick (60g) + fries (125g) | 415g |
| Biryani Meal | Biryani (350g) + chicken (250g) | 600g |
| Sandwich Meal | Sandwich (200g) + fries (125g) | 325g |
| Salad Meal | Carrot (60g) + cucumber (150g) + ¼ onion (25g) + 2 × pepper-chilli (20g) | 255g |

```text
Weight wasted (g) = baseline meal weight × amount wasted (%) / 100
```

#### Edge Cases

If the system detects food but cannot work out which meal it is, it falls back to an **"Unknown Meal"** with the default **25-pixel padding** instead of crashing. Because there is no baseline weight for an unknown meal, its wasted weight is recorded as 0g.

### 2. How I See This Solution Running in Production

In a live cafeteria, percentages alone are not enough; managers need the physical weight to calculate the money lost. Because the gram values are saved to a CSV file automatically, managers could load the data into Excel at the end of the week and see how many kilograms of each meal were thrown away. This can help them adjust their grocery orders and reduce food waste.

That said, estimating weight from a picture is not very reliable. In production, I would suggest placing the bin on a **weighing scale** whose display is visible to the camera. The system could then read the increase in total weight caused by each waste event. My current strategy would serve as a fallback when the scale reading is not available.

### 3. What I Could Have Done Better

- **Ingredient sizes:** Items like carrots and cucumbers come in many sizes, and their weight varies by region. I could have gone to a store and measured the average weight of these items myself.
- **Portion sizes:** The method assumes every portion served weighs the same. For example, it assumes every KFC meal starts at the same base weight, but in reality serving sizes vary.
- **Confidence score:** The confidence score in the output CSV is the average of the YOLO bounding-box confidences in the snapshot frame. Ideally, it should also reflect the confidence of the amount and weight formulas, so a low-quality estimate would show up as a low score.

---

## How to Run the Project

These steps generate the deliverables: the output video, the event images, and the `wastage_log.csv` file.

### 1. Get the Code

Open your terminal and clone the repository:

```bash
git clone https://github.com/ryder-4/food-waste-detection.git
cd food-waste-detection
```

The trained model weights are included at `app/weights/best.pt`.

### 2. Add the Input Video

The `data/` folder is not tracked in git, so create it and place the test video inside:

```bash
mkdir data
# copy your video into data/, e.g. data/IMG_3304.MOV
```

Supported formats are `.mov`, `.mp4`, `.avi` and `.mkv`. If there are several videos, the first one found is used.

### 3. Run the Pipeline

**Option A: Docker (recommended)**

```bash
docker compose up --build
```

The `app/`, `data/` and `output/` folders are mounted into the container, so the results appear in your local `output/` folder.

**Option B: Local Python (3.10)**

```bash
pip install -r app/requirements.txt
cd app
python main.py
```

Run the script from inside `app/`, because the default paths (`../data`, `../output`, `weights/best.pt`) are relative to that folder. You can override them:

```bash
python main.py --video ../data/IMG_3304.MOV --weights weights/best.pt --out_dir ../output
```

### 4. Outputs

All outputs are written to `output/`:

| File | Description |
|------|-------------|
| `pipeline_output.mp4` | Annotated video showing detections, the ROI, the current state and the waste figures |
| `event_<id>_clean_frame.jpg` | The snapshot used for the calculations |
| `event_<id>_amount_wasted.jpg` | Food (green) vs. empty plate (blue) inside the inner ellipse |
| `event_<id>_weight_wasted.jpg` | Detected food items, classified meal and wasted weight |
| `wastage_log.csv` | One row per event: start/end timestamps and frames, classified meal, amount wasted (%), weight wasted (g) and confidence score |

Sample `wastage_log.csv` from the provided video:

| Event_ID | Start (s) | End (s) | Meal | Amount Wasted (%) | Weight Wasted (g) | Confidence |
|---|---|---|---|---|---|---|
| 1 | 7.41 | 29.24 | Biryani Meal | 50.75 | 304.52 | 0.92 |
| 2 | 34.69 | 43.79 | KFC Meal | 43.88 | 182.11 | 0.91 |
| 3 | 49.55 | 56.79 | Sandwich Meal | 53.27 | 173.11 | 0.91 |

---

## References

1. KFC menu item weights – Instagram: [ADD LINK]
2. [ADD SOURCE for biryani / chicken weights]
3. [ADD SOURCE for sandwich weight]
4. [ADD SOURCE for carrot / cucumber / onion / pepper-chilli weights]
