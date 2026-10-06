import json
import os

# Dynamically resolve the JSON path to be relative to this file inside the 'app' folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CURRENT_DIR, 'meal_weights.json')

# Load weights and definitions directly from the JSON file
with open(JSON_PATH, 'r') as f:
    MEAL_DATA = json.load(f)
COMPONENTS_WEIGHTS = MEAL_DATA["components"]
MEAL_DEFINITIONS = MEAL_DATA["meals"]

def classify_meal(detected_food_names):
    """
    Takes a list of string class names detected by YOLO and classifies 
    them into a predefined meal type. Returns the meal name, the inner-circle 
    shrink pad required for that meal, and the baseline total weight.
    """
    salad_components = {"carrot", "onion", "cucumber", "pepper-chilli"}
    detected_salad = salad_components.intersection(set(detected_food_names))
    
    meal_name = "Unknown Meal"
    shrink_pad = 25  # Default safety padding
    
    if "burger" in detected_food_names or "drumstick" in detected_food_names:
        meal_name = "KFC Meal"
        shrink_pad = 50
    elif "sandwich" in detected_food_names:
        meal_name = "Sandwich Meal"
        shrink_pad = 50
    elif len(detected_salad) >= 2:
        meal_name = "Salad Meal"
        shrink_pad = 25
    elif "biryani" in detected_food_names or "chicken" in detected_food_names:
        meal_name = "Biryani Meal"
        shrink_pad = 25

    # Calculate total baseline weight based on classification
    total_meal_weight = 0
    
    if meal_name == "Salad Meal":
        # Specific formula for Salad: 1 carrot, 1 cucumber, 1/4th onion, 2 peppers/chilli
        w_carrot = COMPONENTS_WEIGHTS.get("carrot", 0)
        w_cucumber = COMPONENTS_WEIGHTS.get("cucumber", 0)
        w_onion = COMPONENTS_WEIGHTS.get("onion", 0)
        w_pepper = COMPONENTS_WEIGHTS.get("pepper-chilli", 0)
        
        total_meal_weight = (1 * w_carrot) + (1 * w_cucumber) + (0.25 * w_onion) + (2 * w_pepper)
        
    elif meal_name in MEAL_DEFINITIONS:
        # Standard sum for all other meals
        total_meal_weight = sum([COMPONENTS_WEIGHTS[comp] for comp in MEAL_DEFINITIONS[meal_name]])
        
    return meal_name, shrink_pad, total_meal_weight

def calculate_wasted_weight(total_meal_weight, waste_pct):
    """Applies the calculated waste percentage to the meal's baseline weight."""
    return total_meal_weight * (waste_pct / 100.0)