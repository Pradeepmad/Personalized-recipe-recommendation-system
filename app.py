from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import json
import os
import requests
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
from markupsafe import Markup
import sys

# -----------------------------
# 1. Initialize Flask app
# -----------------------------
app = Flask(__name__)
# IMPORTANT: Use a complex, randomly generated secret key in a real application
app.secret_key = "supersecretkey_gourmetguide"

# NEW: Register the tojson filter for Jinja
def tojson_filter(data):
    """Safely converts Python data to JSON string for use in JavaScript within HTML attributes."""
    
    def custom_serializer(obj):
        # Handle numpy data types
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle date objects from pandas resampling
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    json_string = json.dumps(data, default=custom_serializer)
    # Escape quotes for safe placement within an HTML attribute
    escaped_json = json_string.replace('"', '&quot;') 
    return Markup(escaped_json)

app.jinja_env.filters['tojson'] = tojson_filter
# ---

# --- Load environment variables using the explicit path ---
# Note: Using r"..." for raw string to handle Windows path backslashes
# Replace this path with your actual path if running locally outside the platform
dotenv_path = r"C:\Users\giriv\Desktop\working Projects\recipe\recipe\.env"
load_dotenv(dotenv_path)
# -----------------------------

# Backendless keys
APP_ID = os.getenv("BACKENDLESS_APP_ID")
API_KEY = os.getenv("BACKENDLESS_API_KEY")
if not APP_ID or not API_KEY:
    print("❌ Error: BACKENDLESS_APP_ID or BACKENDLESS_API_KEY not found in environment variables.")
    BASE_URL = "https://api.backendless.com/DUMMY_APP_ID/DUMMY_API_KEY"
else:
    BASE_URL = f"https://api.backendless.com/{APP_ID}/{API_KEY}"

BASE_HEADERS = {"Content-Type": "application/json"}

# -----------------------------
# 2. Load recipe dataset & BERT Embeddings
# -----------------------------
data = None
BERT_EMBEDDINGS = None
EMBEDDING_ID_MAP = {} 
BERT_PATH = r"C:\Users\giriv\Desktop\working Projects\recipe\recipe\bert_embeddings.npy"

try:
    with open("data.json", "r", encoding="utf-8") as f:
        data_list = json.load(f)
    data = pd.DataFrame(data_list)

    for col in ['ingredients', 'steps']:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: x if isinstance(x, list) else [])
    
    # CRUCIAL TYPE CONVERSION
    if 'recipe_id' in data.columns:
        data['recipe_id'] = data['recipe_id'].astype(str)
        print("✅ Coerced 'data' recipe_id column to string for consistency.")

    print("✅ Loaded dataset:", data.shape)

    # --- Load BERT Embeddings (Logic retained but not used for reporting) ---
    if os.path.exists(BERT_PATH):
        BERT_EMBEDDINGS = np.load(BERT_PATH)
        print(f"✅ Loaded BERT Embeddings: {BERT_EMBEDDINGS.shape}")
        
        if data.shape[0] == BERT_EMBEDDINGS.shape[0]:
            EMBEDDING_ID_MAP = {recipe_id: i for i, recipe_id in enumerate(data['recipe_id'])}
        else:
            BERT_EMBEDDINGS = None
    else:
        pass

except Exception as e:
    print(f"❌ Error during data loading: {e}")
    print("Application will run with dummy data.")
    data = pd.DataFrame([
        {'recipe_id': '1', 'recipe_name': 'Dummy Chicken Stew', 'ingredients': ['Chicken', 'Carrot', 'Water'], 'nutritions': {'calories': 300, 'protein_g': 30, 'carbs_g': 15, 'fat_g': 10}, 'veg_nonveg': 'Non-Veg', 'diet_goal': 'Muscle_Gain', 'health_score': 85, 'image_link': 'https://placehold.co/600x400/FF7D00/FFFFFF?text=Dummy+Recipe', 'steps': ['Cook chicken.', 'Add vegetables.']},
        {'recipe_id': '2', 'recipe_name': 'Dummy Salad', 'ingredients': ['Lettuce', 'Tomato', 'Cucumber'], 'nutritions': {'calories': 150, 'protein_g': 5, 'carbs_g': 10, 'fat_g': 5}, 'veg_nonveg': 'Veg', 'diet_goal': 'Weight_Loss', 'health_score': 95, 'image_link': 'https://placehold.co/600x400/00A388/FFFFFF?text=Dummy+Salad', 'steps': ['Mix all ingredients.']},
        {'recipe_id': '3', 'recipe_name': 'Dummy High Calorie Shake', 'ingredients': ['Milk', 'Peanut Butter', 'Banana'], 'nutritions': {'calories': 700, 'protein_g': 20, 'carbs_g': 70, 'fat_g': 30}, 'veg_nonveg': 'Veg', 'diet_goal': 'Weight_Gain', 'health_score': 70, 'image_link': 'https://placehold.co/600x400/00A388/FFFFFF?text=Weight+Gain+Shake', 'steps': ['Blend it all up.']},
        {'recipe_id': '4', 'recipe_name': 'Allergy Test Recipe', 'ingredients': ['Nuts', 'Wheat', 'Sugar'], 'nutritions': {'calories': 200, 'protein_g': 5, 'carbs_g': 40, 'fat_g': 2}, 'veg_nonveg': 'Veg', 'diet_goal': 'Maintenance', 'health_score': 80, 'image_link': 'https://placehold.co/600x400/00A388/FFFFFF?text=Allergy+Test', 'steps': ['Bake it.']},
    ])
    if 'recipe_id' in data.columns:
          data['recipe_id'] = data['recipe_id'].astype(str)
        
# -----------------------------
# 3. Helper functions
# -----------------------------

def get_recipe_details_by_id(recipe_id):
    """Fetches ALL details for a single recipe from the global DataFrame."""
    if data is None or data.empty:
        return None

    recipe_id = str(recipe_id)

    result = data[data['recipe_id'] == recipe_id]

    if not result.empty:
        recipe = result.iloc[0].to_dict()
        
        for key in ['ingredients', 'steps']:
            if key in recipe and not isinstance(recipe[key], list):
                try:
                    recipe[key] = json.loads(str(recipe[key]).replace("'", '"'))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    recipe[key] = [recipe[key]] if recipe[key] else []

        return recipe
    return None

def generate_diet_report(cooked_recipes, user_profile):
    """
    Processes cooked recipes to generate daily/weekly/monthly nutrition reports 
    and gives feedback based on the user's diet goal. Calculates targets using BMR/TDEE.
    """
    
    # 1. BMR/TDEE Calculation based on User Profile
    age = user_profile.get('age')
    weight = user_profile.get('weight')
    height = user_profile.get('height')
    gender = user_profile.get('gender')
    diet_goal = user_profile.get('diet_goal')
    duration_months = user_profile.get('duration_months', 3) # Default 3 months

    # A. Check for minimum data for calculation
    if not (age and weight and height and gender):
        # Set sensible default targets if calculation data is missing
        target_goals = {
            'goal': diet_goal if diet_goal else 'maintenance',
            'duration_months': duration_months,
            'target_calories': 2000, 'target_protein': 75, 'target_carbs': 250, 'target_fat': 60,
            'target_weekly_calories': 14000, 'target_weekly_protein': 525, 'target_weekly_carbs': 1750, 'target_weekly_fat': 420,
            'target_monthly_calories': 60880, 'target_monthly_protein': 2283, 'target_monthly_carbs': 7610, 'target_monthly_fat': 1826,
            'actual_avg_calories': 0, 'actual_avg_protein': 0, 'actual_avg_carbs': 0, 'actual_avg_fat': 0,
        }
        return {"report": {}, "suggestions": ["⚠️ Please fill in your **Age, Gender, Height, and Weight** in the Edit Profile tab to calculate personalized targets."], "target_goals": target_goals}


    # B. Mifflin-St Jeor BMR Equation
    # Weight in kg, Height in cm, Age in years
    if gender == 'male':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif gender == 'female':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        # Fallback BMR if gender is unknown/unset
        bmr = (10 * weight) + (6.25 * height) - (5 * age) 

    # C. Estimate TDEE (Using a sedentary/lightly active multiplier)
    # TDEE = BMR * Activity Multiplier (1.375 = Lightly Active)
    ACTIVITY_MULTIPLIER = 1.375 
    tdee = bmr * ACTIVITY_MULTIPLIER

    # D. Target Setting (Daily)
    target_calories = tdee
    target_protein_pct, target_carbs_pct, target_fat_pct = 0.30, 0.40, 0.30 # Default Maintenance Split

    suggestions = []

    if diet_goal == 'weight_loss':
        target_calories = max(1200, tdee - 500) # Reduce by 500, with a safety minimum
        target_protein_pct, target_carbs_pct, target_fat_pct = 0.35, 0.40, 0.25
        suggestions.append(f"🎯 **Daily Goal:** Aim for a deficit. Calorie Goal: **{target_calories:.0f} kcal** (from TDEE {tdee:.0f} kcal).")

    elif diet_goal == 'muscle_gain':
        target_calories = tdee + 250 # Calorie surplus for muscle building
        target_protein_pct, target_carbs_pct, target_fat_pct = 0.40, 0.35, 0.25
        suggestions.append(f"🎯 **Daily Goal:** Aim for a surplus. Calorie Goal: **{target_calories:.0f} kcal** (from TDEE {tdee:.0f} kcal).")

    elif diet_goal == 'weight_gain':
        target_calories = tdee + 500
        target_protein_pct, target_carbs_pct, target_fat_pct = 0.30, 0.45, 0.25
        suggestions.append(f"🎯 **Daily Goal:** Aim for a surplus. Calorie Goal: **{target_calories:.0f} kcal** (from TDEE {tdee:.0f} kcal).")
    
    else: # No specific goal / maintenance
        suggestions.append(f"ℹ️ **Maintenance Goal:** Your estimated maintenance calories are **{tdee:.0f} kcal**.")


    # Calculate Macro Targets (1g Protein/Carbs = 4 kcal, 1g Fat = 9 kcal)
    target_protein = (target_calories * target_protein_pct) / 4
    target_carbs = (target_calories * target_carbs_pct) / 4
    target_fat = (target_calories * target_fat_pct) / 9

    # E. Structure Targets (Daily, Weekly, Monthly)
    target_goals = {
        'goal': diet_goal if diet_goal else 'maintenance',
        'duration_months': duration_months,
        
        # Daily Targets
        'target_calories': target_calories,
        'target_protein': target_protein,
        'target_carbs': target_carbs,
        'target_fat': target_fat,
        
        # Weekly Targets (Daily * 7)
        'target_weekly_calories': target_calories * 7,
        'target_weekly_protein': target_protein * 7,
        'target_weekly_carbs': target_carbs * 7,
        'target_weekly_fat': target_fat * 7,
        
        # Monthly Targets (Daily * 30.44 avg days)
        'target_monthly_calories': target_calories * 30.44,
        'target_monthly_protein': target_protein * 30.44,
        'target_monthly_carbs': target_carbs * 30.44,
        'target_monthly_fat': target_fat * 30.44,
    }


    # 2. Process history and generate report
    if not cooked_recipes:
        target_goals.update({
             'actual_avg_calories': 0, 'actual_avg_protein': 0, 'actual_avg_carbs': 0, 'actual_avg_fat': 0
        })
        return {"report": {}, "suggestions": suggestions + ["Start logging meals to see your diet history!"], "target_goals": target_goals}
        
    report_data = []
    
    for item in cooked_recipes:
        try:
            date_cooked = datetime.fromisoformat(item['date_cooked']).date()
            nutritions = item.get('nutritions', {})
            
            # Explicitly cast values to float for robust calculation
            report_data.append({
                'date': date_cooked,
                'calories': float(nutritions.get('calories', 0.0)),
                'protein': float(nutritions.get('protein_g', 0.0)),
                'carbs': float(nutritions.get('carbs_g', 0.0)),
                'fat': float(nutritions.get('fat_g', 0.0))
            })
        except Exception as e:
            print(f"Skipping history item due to parsing error: {e}")
            continue

    if not report_data:
        return {"report": {}, "suggestions": suggestions + ["No valid meal data found for reporting."], "target_goals": target_goals}
        
    df = pd.DataFrame(report_data)
    
    # Calculate daily, weekly, and monthly totals using pandas resampling
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Calculate sums, replace 0 with NaN, drop fully empty days, fill remaining NaN with 0
    daily = df.resample('D').sum().replace(0, np.nan).dropna(how='all').fillna(0)
    weekly = df.resample('W').sum().replace(0, np.nan).dropna(how='all').fillna(0)
    monthly = df.resample('M').sum().replace(0, np.nan).dropna(how='all').fillna(0)
    
    report = {
        "daily": daily.reset_index().to_dict('records'),
        "weekly": weekly.reset_index().to_dict('records'),
        "monthly": monthly.reset_index().to_dict('records'),
    }

    # 3. Calculate Actual Averages and provide final suggestions
    logged_days = daily[daily['calories'] > 0]
    
    if not logged_days.empty:
        avg_calories = logged_days['calories'].mean()
        avg_protein = logged_days['protein'].mean()
        avg_carbs = logged_days['carbs'].mean()
        avg_fat = logged_days['fat'].mean()
    else:
        avg_calories, avg_protein, avg_carbs, avg_fat = 0, 0, 0, 0

    # Update target_goals with actual averages
    target_goals.update({
        'actual_avg_calories': avg_calories,
        'actual_avg_protein': avg_protein,
        'actual_avg_carbs': avg_carbs,
        'actual_avg_fat': avg_fat,
    })

    # Additional suggestions based on actual intake vs. target
    if avg_calories > target_calories * 1.10:
        suggestions.append(f"⚠️ **Calorie Warning:** Your average intake ({avg_calories:.0f} kcal) is **above** your goal ({target_calories:.0f} kcal). Focus on smaller, nutrient-dense portions.")
    elif avg_calories < target_calories * 0.90:
        suggestions.append(f"⚠️ **Calorie Warning:** Your average intake ({avg_calories:.0f} kcal) is **below** your goal ({target_calories:.0f} kcal). Consider adding a high-quality snack.")
    else:
        suggestions.append("✅ **Calorie Status:** Your average daily calorie intake is within 10% of your target. Great consistency!")
    
    if avg_protein < target_protein * 0.90:
        suggestions.append(f"📈 **Protein Boost:** Your average protein ({avg_protein:.0f}g) is below the goal ({target_protein:.0f}g). Ensure protein at every meal.")

    return {"report": report, "suggestions": suggestions, "target_goals": target_goals}


def search_by_ingredients(ingredients_str, top_n=10):
    """Filters the dataset based on provided ingredients."""
    if data is None or data.empty:
        return pd.DataFrame()

    ingredients = [i.strip().lower() for i in ingredients_str.split(",") if i.strip()]
    if not ingredients:
        return pd.DataFrame()

    def contains_all(recipe_ings):
        """Checks if all search ingredients are present in the recipe's ingredients (partial match)."""
        if isinstance(recipe_ings, list):
            recipe_ings_lower = [str(i).lower() for i in recipe_ings]
            return all(any(ing in recipe_ing for recipe_ing in recipe_ings_lower) for ing in ingredients)
        return False

    filtered = data[data['ingredients'].apply(contains_all)]

    if 'health_score' in filtered.columns:
        filtered = filtered.sort_values(by='health_score', ascending=False)

    return filtered.head(top_n)

def get_user_profile(user_id):
    """Fetches user profile data from Backendless, including new fields."""
    if not user_id:
        return None

    url = f"{BASE_URL}/data/UserProfile?where=user_id='{user_id}'"
    try:
        r = requests.get(url, headers=BASE_HEADERS)

        if r.status_code == 200 and r.json():
            profile = r.json()[0]
            # Process allergies: split CSV string into a list of lowercase, stripped strings
            if profile.get("allergies"):
                profile["allergies"] = [a.strip().lower() for a in profile["allergies"].split(',') if a.strip()]
            else:
                profile["allergies"] = []
            
            # Safely convert strings/floats to numbers if they exist
            profile['age'] = int(profile.get('age')) if profile.get('age') else None
            profile['height'] = float(profile.get('height')) if profile.get('height') else None
            profile['weight'] = float(profile.get('weight')) if profile.get('weight') else None
            profile['duration_months'] = int(profile.get('duration_months')) if profile.get('duration_months') else 3 # Default to 3 months

            return profile
        elif r.status_code == 200 and not r.json():
            return None
        else:
            print(f"Error fetching user profile (Status {r.status_code}): {r.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching user profile (Network): {e}")
    return None

def get_recipe_review(user_id, recipe_id):
    """Fetches the user's existing review (rating and comment) for a specific recipe."""
    if not user_id or not recipe_id:
        return None
    
    where_clause = f"user_id='{user_id}' and recipe_id='{recipe_id}'"
    url = f"{BASE_URL}/data/RecipeReviews?where={where_clause}"
    
    try:
        r = requests.get(url, headers=BASE_HEADERS)
        if r.status_code == 200 and r.json():
            return r.json()[0]
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching existing review: {e}")
        return None

def get_cooked_recipes(user_id):
    """
    Fetches the history of recipes and explicitly fetches all details 
    for each recipe ID from local data.
    """
    if not user_id:
        return []

    # 1. Fetch Cooked Recipes (History) from Backendless
    url = f"{BASE_URL}/data/CookedRecipes?where=user_id='{user_id}'&sortBy=date_cooked%20desc"
    try:
        r = requests.get(url, headers=BASE_HEADERS)
        if r.status_code != 200:
            print(f"Error fetching cooked recipes (Status {r.status_code}): {r.text}")
            return []
        
        cooked_history = r.json()
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching cooked recipes: {e}")
        return []

    if not cooked_history:
        return []

    history_list = []
    
    for history_entry in cooked_history:
        recipe_id = str(history_entry.get('recipe_id'))
        
        # 2. Lookup full recipe details and nutrition data locally (from data.json)
        recipe_details = get_recipe_details_by_id(recipe_id)

        if recipe_details:
            recipe_name = recipe_details.get('recipe_name', 'Unknown Recipe (No Name)')
            
            # 3. Fetch review status
            review_data = get_recipe_review(user_id, recipe_id)

            history_item = {
                "date_cooked": history_entry.get('date_cooked'),
                "recipe_name": recipe_name, 
                "image_link": recipe_details.get('image_link', 'https://placehold.co/80x80/FF7D00/FFFFFF?text=Recipe'),
                "recipe_id": recipe_id,
                "nutritions": recipe_details.get('nutritions', {}),
                "veg_nonveg": recipe_details.get('veg_nonveg', ''),
                "ingredients": recipe_details.get('ingredients', []),
                "steps": recipe_details.get('steps', []),
                "review": review_data
            }
        else:
            recipe_name = f"Unknown Recipe (ID: {recipe_id} - MISSING LOCAL DATA)"
            print(f"❌ Missing data for Recipe ID: {recipe_id}")

            review_data = get_recipe_review(user_id, recipe_id)
            
            history_item = {
                "date_cooked": history_entry.get('date_cooked'),
                "recipe_name": recipe_name, 
                "image_link": 'https://placehold.co/80x80/FF7D00/FFFFFF?text=Recipe',
                "recipe_id": recipe_id,
                "nutritions": {},
                "veg_nonveg": '',
                "ingredients": [],
                "steps": [],
                "review": review_data
            }

        history_list.append(history_item)
            
    return history_list


def get_recipes_data(results, user_profile=None):
    """Formats DataFrame rows into a list of recipe dictionaries for the template."""
    user_allergies = user_profile.get("allergies", []) if user_profile and isinstance(user_profile, dict) else []

    recipes = []
    if isinstance(results, pd.DataFrame):
        for _, row in results.iterrows():
            # Ensure all fields are present with fallbacks
            recipes.append({
                "recipe_id": str(row.get('recipe_id', '')),
                "recipe_name": row.get('recipe_name', 'Unnamed Recipe'),
                "ingredients": row.get('ingredients', []),
                "nutritions": row.get('nutritions', {}),
                "veg_nonveg": row.get('veg_nonveg', 'Both'),
                "diet_goal": row.get('diet_goal', 'Maintenance'),
                "health_score": row.get('health_score', 0),
                "image_link": row.get('image_link', 'https://placehold.co/600x400/999999/FFFFFF?text=No+Image'),
                "steps": row.get('steps', []),
            })
    return recipes, user_allergies

def suggest_daily_recipes(user_profile, top_n=3):
    """
    Suggests recipes based on user profile (diet_goal, food_pref, allergies).
    Prioritizes recipes that align with the diet goal and excludes allergens.
    """
    global data
    if data is None or data.empty:
        return []

    # 1. Get User Preferences
    # --- FIX for AttributeError: 'NoneType' object has no attribute 'lower' ---
    # We must explicitly check the result of .get() to prevent errors if the key exists but holds a None value.
    
    # Handle diet_goal defensively
    raw_diet_goal = user_profile.get("diet_goal")
    if raw_diet_goal is None:
        diet_goal = "maintenance"
    else:
        diet_goal = str(raw_diet_goal).lower()
    
    # Handle food_pref defensively
    raw_food_pref = user_profile.get("food_pref")
    if raw_food_pref is None:
        food_pref = "both"
    else:
        food_pref = str(raw_food_pref).lower()
    
    user_allergies = user_profile.get("allergies", [])
    
    filtered_data = data.copy()

    # --- Initial Data Prep & Filtering ---
    # Prepare nutrition columns for sorting, providing default safe values
    if 'nutritions' in filtered_data.columns:
        filtered_data['calories'] = filtered_data['nutritions'].apply(lambda x: x.get('calories', 0.0) if isinstance(x, dict) else 0.0)
        filtered_data['protein_g'] = filtered_data['nutritions'].apply(lambda x: x.get('protein_g', 0.0) if isinstance(x, dict) else 0.0)
    else:
        # Create dummy columns if missing, to prevent errors
        filtered_data['calories'] = 0.0
        filtered_data['protein_g'] = 0.0

    # 2. Filter by Food Preference (Veg/Non-Veg)
    if food_pref in ["veg", "non-veg"]:
        filtered_data = filtered_data[filtered_data['veg_nonveg'].str.lower() == food_pref]

    if filtered_data.empty:
        return []

    # 3. Exclude by Allergies
    if user_allergies:
        def contains_allergen(recipe_ings):
            """Checks if any recipe ingredient contains a user's allergen (partial match)."""
            if isinstance(recipe_ings, list):
                recipe_ings_lower = [str(i).lower() for i in recipe_ings]
                # Returns True if any allergen is found in any ingredient
                return any(any(allergen in recipe_ing for recipe_ing in recipe_ings_lower) for allergen in user_allergies)
            return False
            
        # Keep recipes where 'contains_allergen' is False
        filtered_data = filtered_data[~filtered_data['ingredients'].apply(contains_allergen)]

    if filtered_data.empty:
        return []

    # 4. Sort by Diet Goal (with health_score as secondary tie-breaker)
    sort_criteria = []
    ascending_order = []
    
    if diet_goal == "weight_loss":
        # Lowest calories first
        sort_criteria = ['calories', 'health_score']
        ascending_order = [True, False] 
    elif diet_goal == "muscle_gain":
        # Highest protein first
        sort_criteria = ['protein_g', 'health_score']
        ascending_order = [False, False] 
    elif diet_goal == "weight_gain":
        # Highest calories first
        sort_criteria = ['calories', 'health_score']
        ascending_order = [False, False] 
    else:
        # Maintenance/Other: Fallback to best health score
        sort_criteria = ['health_score']
        ascending_order = [False]
        
    # Check if 'health_score' exists before trying to sort by it
    if 'health_score' not in filtered_data.columns:
        if 'health_score' in sort_criteria:
            sort_criteria.remove('health_score')
            ascending_order = ascending_order[:-1] # Remove corresponding boolean

    # Perform the sort
    if sort_criteria:
        filtered_data = filtered_data.sort_values(by=sort_criteria, ascending=ascending_order)

    # 5. Return the top N recipes
    top_recipes = filtered_data.head(top_n)
    
    recipes_list, _ = get_recipes_data(top_recipes, user_profile)
    
    return recipes_list


# -----------------------------
# 4. Authentication and Home Routes (UPDATED /dashboard)
# -----------------------------
@app.route("/")
def home():
    """Renders the main landing page (main.html)."""
    if "user" in session:
        return redirect(url_for("dashboard"))
    # Note: main.html and index.html are assumed to exist and are not generated here.
    return render_template("main.html", message=request.args.get('message'))

@app.route("/dashboard")
def dashboard():
    """Renders the recipe grid (index.html) with personalized data."""
    if "user" not in session:
        return redirect(url_for("home", message="⚠️ Please log in to view the dashboard."))

    user_id = session["user"].get("objectId")
    user_profile = get_user_profile(user_id) # Ensure this is fetched

    if data is None or data.empty:
        return render_template("index.html", recipes=[], user_allergies=[], message="❌ Recipe data is unavailable.", session=session)

    # --- Fetch Daily Suggestions ---
    daily_suggestions = suggest_daily_recipes(user_profile, top_n=3)
    
    if daily_suggestions:
        # Use suggestions as the main list if available
        recipes = daily_suggestions
        # Retrieve the corrected diet goal string for display purposes
        raw_diet_goal = user_profile.get("diet_goal")
        diet_goal_str = str(raw_diet_goal).lower() if raw_diet_goal is not None else "maintenance"
        goal_name = diet_goal_str.replace('_', ' ').title()
        
        suggestion_message = f"Your daily personalized recommendations for **{goal_name}** goal."
        # To correctly display user_allergies, we extract them from the fetched profile
        user_allergies = user_profile.get("allergies", [])
    else:
        # Fallback to popular recipes
        if 'health_score' in data.columns:
            popular_recipes = data.sort_values(by='health_score', ascending=False).head(10)
        else:
            popular_recipes = data.head(10)
        
        recipes, user_allergies = get_recipes_data(popular_recipes, user_profile)
        suggestion_message = "Popular Recipes Today (No profile set or no matching recipes found)"
    # -------------------------------

    return render_template("index.html",
                           recipes=recipes,
                           user_allergies=user_allergies,
                           message=request.args.get('message', suggestion_message),
                           session=session)


@app.route("/signup", methods=["POST"])
def signup():
    """Handles user registration via Backendless and immediately initializes UserProfile."""
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    payload = {"name": name, "email": email, "password": password}
    url = f"{BASE_URL}/users/register"
    
    try:
        r = requests.post(url, json=payload, headers=BASE_HEADERS)
        
        if r.status_code == 200:
            new_user_data = r.json()
            new_user_id = new_user_data.get("objectId")
            
            # Initialize UserProfile with user_id
            profile_payload = {"user_id": new_user_id}
            profile_url = f"{BASE_URL}/data/UserProfile"
            
            try:
                profile_r = requests.post(profile_url, json=profile_payload, headers=BASE_HEADERS)
                if profile_r.status_code not in [200, 201]:
                    print(f"⚠️ Warning: Failed to initialize UserProfile for {new_user_id}. Status: {profile_r.status_code}, Error: {profile_r.text}")
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Warning: Network error during UserProfile initialization for {new_user_id}: {e}")
                
            return redirect(url_for("home", message="✅ Signup successful! Please login."))
        else:
            error_message = r.json().get('message', 'Unknown Backendless Error') if r.json() else 'Unknown Error'
            print("Signup error:", r.text)
            return redirect(url_for("home", message=f"❌ Signup failed. Reason: {error_message}"))
            
    except requests.exceptions.RequestException:
        return redirect(url_for("home", message="❌ Network error during signup."))

@app.route("/login", methods=["POST"])
def login():
    """Handles user login via Backendless."""
    email = request.form.get("email")
    password = request.form.get("password")

    payload = {"login": email, "password": password}
    url = f"{BASE_URL}/users/login"
    try:
        r = requests.post(url, json=payload, headers=BASE_HEADERS)
        if r.status_code == 200:
            user = r.json()
            session["user"] = user
            return redirect(url_for("dashboard"))
        else:
            print("Login error:", r.text)
            return redirect(url_for("home", message="❌ Invalid credentials or user not found."))
    except requests.exceptions.RequestException:
        return redirect(url_for("home", message="❌ Network error during login."))

@app.route("/logout")
def logout():
    """Clears the user session."""
    session.clear()
    return redirect(url_for("home", message="You have been logged out."))

# -----------------------------
# 5. Profile Route (Updated for new fields and TDEE calculation)
# -----------------------------
@app.route("/profile", methods=["GET", "POST"])
def profile():
    """Handles viewing and updating the user profile and generates diet reports."""
    if "user" not in session:
        return redirect(url_for("home"))

    user = session["user"]
    user_id = user.get("objectId")
    # Fetch latest profile data
    user_profile = get_user_profile(user_id)
    if user_profile is None:
        user_profile = {}


    message = request.args.get('message')
    active_tab = request.args.get('active_tab', 'v-pills-profile')
    
    diet_report = None
    cooked_recipes = None

    if request.method == "POST":
        # Handle profile update logic
        try:
            # New fields added to the form
            age = int(request.form.get("age")) if request.form.get("age") else None
            duration_months = int(request.form.get("duration_months")) if request.form.get("duration_months") else 3
            height = float(request.form.get("height")) if request.form.get("height") else None
            weight = float(request.form.get("weight")) if request.form.get("weight") else None
            
        except ValueError:
            message = "❌ Height, Weight, Age, and Duration must be valid numbers."
            active_tab = 'v-pills-edit'
            return render_template("profile.html", profile=user_profile, message=message, active_tab=active_tab, session=session)

        allergies = request.form.get("allergies", "")
        diet_goal = request.form.get("diet_goal")
        food_pref = request.form.get("food_pref")
        gender = request.form.get("gender") # NEW

        payload = {
            "user_id": user_id,
            "height": height,
            "weight": weight,
            "allergies": allergies,
            "diet_goal": diet_goal,
            "food_pref": food_pref,
            # New fields for calculation
            "age": age,
            "gender": gender,
            "duration_months": duration_months
        }

        # Save logic remains the same (PUT or POST to Backendless)
        if user_profile and "objectId" in user_profile:
            object_id = user_profile["objectId"]
            url = f"{BASE_URL}/data/UserProfile/{object_id}"
            r = requests.put(url, json=payload, headers=BASE_HEADERS)
        else:
            url = f"{BASE_URL}/data/UserProfile"
            r = requests.post(url, json=payload, headers=BASE_HEADERS)

        if r.status_code in [200, 201]:
            # Refetch the profile to get the objectId and updated values for the return
            user_profile = get_user_profile(user_id) 
            return redirect(url_for("profile", message="✅ Profile updated successfully!", active_tab='v-pills-profile'))
        else:
            print("Profile save error:", r.text)
            message = "❌ Failed to save profile."
            active_tab = 'v-pills-edit'
            # Update user_profile with attempted payload data in case of error
            user_profile.update(payload)
    
    # --- Data Fetching for Tabs ---
    if active_tab == 'v-pills-history' or active_tab == 'v-pills-diet-history':
        # Fetch recipes only once if needed for either history or report
        cooked_recipes = get_cooked_recipes(user_id) 
        
    if active_tab == 'v-pills-diet-history':
        # Pass the latest user_profile data to the report generator
        diet_report = generate_diet_report(cooked_recipes, user_profile)
    # --- End Data Fetching ---

    return render_template("profile.html",
                           profile=user_profile,
                           cooked_recipes=cooked_recipes,
                           diet_report=diet_report,
                           message=message,
                           active_tab=active_tab,
                           session=session)

# -----------------------------
# 6. Recipe Recommendation
# -----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    """Handles the ingredient-based recipe search and filtering."""
    if "user" not in session:
        return redirect(url_for("home", message="⚠️ Please log in to search for recipes."))

    if data is None or data.empty:
        return render_template("index.html", recipes=[], user_allergies=[], message="❌ Recipe data is unavailable for search.", session=session)

    ingredients = request.form.get("query")
    if not ingredients:
        return redirect(url_for("dashboard", message="⚠️ Please enter at least one ingredient."))

    results = search_by_ingredients(ingredients)

    if results.empty:
        return render_template("index.html", recipes=[], user_allergies=[], message=f"❌ No recipes found containing all of the ingredients: '{ingredients}'.", session=session)

    user_id = session["user"].get("objectId")
    user_profile = get_user_profile(user_id)

    # Apply filtering/sorting based on user profile
    if user_profile:
        food_pref_raw = user_profile.get("food_pref")
        food_pref = str(food_pref_raw).lower() if food_pref_raw is not None else "both"
        
        diet_goal_raw = user_profile.get("diet_goal")
        diet_goal = str(diet_goal_raw).lower() if diet_goal_raw is not None else ""

        # 2. Food Preference Filter (Veg/Non-Veg)
        if food_pref in ["veg", "non-veg"]:
            results = results[results['veg_nonveg'].str.lower() == food_pref]

        if results.empty:
            return render_template("index.html", recipes=[], message=f"❌ No recipes found matching your ingredients AND your dietary preferences (Food Pref: {food_pref.capitalize()}).", session=session)

        # 3. Diet Goal Sorting (Lower Calorie for weight loss, higher protein for muscle gain, etc.)
        if diet_goal == "weight_loss":
            results['calories'] = results['nutritions'].apply(lambda x: x.get('calories', float('inf')) if isinstance(x, dict) else float('inf'))
            results = results.sort_values(by='calories', ascending=True)
        elif diet_goal == "muscle_gain":
            results['protein_g'] = results['nutritions'].apply(lambda x: x.get('protein_g', float('-inf')) if isinstance(x, dict) else float('-inf'))
            results = results.sort_values(by='protein_g', ascending=False)
        elif diet_goal == "weight_gain":
            results['calories'] = results['nutritions'].apply(lambda x: x.get('calories', float('-inf')) if isinstance(x, dict) else float('-inf'))
            results = results.sort_values(by='calories', ascending=False)

    recipes, user_allergies = get_recipes_data(results, user_profile)

    return render_template("index.html", recipes=recipes, user_allergies=user_allergies, message=f"✅ Found {len(recipes)} recipes matching: {ingredients}", session=session)


# -----------------------------
# 7. New API Route for Daily Suggestions
# -----------------------------
@app.route("/api/daily_suggestions", methods=["GET"])
def api_daily_suggestions():
    """
    API route to fetch daily recipe suggestions based on user profile.
    Returns JSON response for dynamic loading on the dashboard.
    """
    if "user" not in session:
        return jsonify({"success": False, "message": "Authentication required."}), 401
    
    user_id = session["user"].get("objectId")
    user_profile = get_user_profile(user_id)
    
    if not user_profile:
        return jsonify({"success": False, "message": "Profile data missing. Please update your profile (Age, Weight, Height, Diet Goal) for personalized suggestions."}), 400

    # Get the top 3 suggestions
    suggestions = suggest_daily_recipes(user_profile, top_n=3)

    if not suggestions:
        message = "Could not find any matching recipes after applying your profile filters (Food Preference and Allergies)."
    else:
        message = "Daily Personalized Recipe Suggestions"

    return jsonify({
        "success": True, 
        "message": message,
        "suggestions": suggestions
    })


# -----------------------------
# 8. Mark as Eaten / Review / History Routes
# -----------------------------

@app.route("/mark_eaten/<recipe_id>", methods=["POST"])
def mark_eaten(recipe_id):
    """Records a recipe as cooked/eaten by the current user."""
    if "user" not in session:
        return jsonify({"success": False, "message": "❌ Please log in to mark a recipe as eaten."}), 401

    user_id = session["user"].get("objectId")

    payload = {
        "user_id": user_id,
        "recipe_id": recipe_id,
        "date_cooked": datetime.now().isoformat()
    }

    url = f"{BASE_URL}/data/CookedRecipes"
    try:
        r = requests.post(url, json=payload, headers=BASE_HEADERS)
        if r.status_code in [200, 201]:
            review_exists = get_recipe_review(user_id, recipe_id) is not None
            return jsonify({"success": True, "message": "✅ Recipe marked as cooked! Ready to review.", "review_exists": review_exists})
        else:
            error_message = r.json().get('message', 'Unknown Backendless Error') if r.json() else 'Unknown Error'
            print("Mark eaten error:", r.text)
            return jsonify({"success": False, "message": f"⚠️ Failed to save recipe history. Reason: {error_message}"}), 500
    except requests.exceptions.RequestException:
        return jsonify({"success": False, "message": "❌ Network error during save."}), 500

@app.route("/review/<recipe_id>", methods=["POST"])
def review_recipe(recipe_id):
    """Saves a user's rating and comment for a recipe to the Backendless RecipeReviews table."""
    if "user" not in session:
        return jsonify({"success": False, "message": "❌ Please log in to submit a review."}), 401

    user_id = session["user"].get("objectId")

    try:
        data = request.get_json()
        rating = int(data.get("rating"))
        comment = data.get("comment", "").strip()

        if rating < 1 or rating > 5:
            return jsonify({"success": False, "message": "❌ Invalid rating value."}), 400

    except Exception as e:
        print(f"Review data parsing error: {e}")
        return jsonify({"success": False, "message": "❌ Invalid data format for review."}), 400

    payload = {
        "user_id": user_id,
        "recipe_id": recipe_id,
        "rating": rating,
        "comment": comment,
        "date_reviewed": datetime.now().isoformat()
    }

    url = f"{BASE_URL}/data/RecipeReviews"
    try:
        r = requests.post(url, json=payload, headers=BASE_HEADERS)

        if r.status_code in [200, 201]:
            return jsonify({"success": True, "message": f"⭐ Thank you for your {rating}-star review! Your feedback is saved."})
        else:
            error_message = r.json().get('message', 'Unknown Backendless Error') if r.json() else 'Unknown Error'
            print("Review save error:", r.text)
            return jsonify({"success": False, "message": f"⚠️ Failed to save review. Reason: {error_message}"}), 500

    except requests.exceptions.RequestException:
        return jsonify({"success": False, "message": "❌ Network error during review submission."}), 500

@app.route("/history")
def history():
    """Forces the history tab open by redirecting to the profile route."""
    if "user" not in session:
        return redirect(url_for("home"))

    return redirect(url_for('profile', active_tab='v-pills-history'))


# -----------------------------
# 9. Run Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
