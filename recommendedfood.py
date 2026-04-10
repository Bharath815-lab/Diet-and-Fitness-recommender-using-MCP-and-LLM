from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

# Config - update paths if needed
CSV_PATH = os.getenv("NUTRITION_CSV_PATH",
                     r"/home/deepak/Downloads/fitness calorie (3)(1)/fitness calorie (2)/fitness calorie/nutrition_updated.csv")

app = Flask(__name__)
CORS(app)

# Load food data once
df = pd.read_csv(CSV_PATH)
# Ensure numeric columns and drop invalid rows
df['calories'] = pd.to_numeric(df.get('calories', 0), errors='coerce').fillna(0)
df['protein'] = pd.to_numeric(df.get('protein', 0), errors='coerce').fillna(0)
df = df[df['calories'] > 0].reset_index(drop=True)  # keep only positive-calorie foods

def calculate_new_weight(current_weight, total_calories):
    weight_change = total_calories / 7700.0
    new_weight = current_weight - weight_change
    return new_weight

def recommend_foods(predicted_calories, weight_difference_percentage,
                    num_sets=3, items_per_set=3, max_rows=None):
    """
    Returns up to num_sets recommendation sets. Each set contains exactly items_per_set items.
    Optimization objective: maximize total protein while keeping total calories within [min_calories, max_calories].
    """
    if max_rows is not None:
        food_df = df.head(max_rows).copy()
    else:
        food_df = df.copy()

    min_calories = (80 + weight_difference_percentage) * predicted_calories / 100.0
    max_calories = (100 + weight_difference_percentage) * predicted_calories / 100.0

    # Guard: if min_calories > max_calories, swap
    if min_calories > max_calories:
        min_calories, max_calories = max_calories, min_calories

    solutions = []
    excluded_indices = set()

    for _ in range(num_sets):
        problem = LpProblem("Maximize_Protein", LpMaximize)
        n = len(food_df)
        choices = [LpVariable(f"choice_{i}", cat="Binary") for i in range(n)]

        # Objective: maximize protein
        problem += lpSum(choices[i] * float(food_df.iloc[i]['protein']) for i in range(n))

        # Calorie constraints
        total_cal = lpSum(choices[i] * float(food_df.iloc[i]['calories']) for i in range(n))
        problem += total_cal >= min_calories
        problem += total_cal <= max_calories

        # Exact number of items
        problem += lpSum(choices) == items_per_set

        # Exclude already-chosen items by fixing choice variable to 0
        for idx in excluded_indices:
            if 0 <= idx < n:
                problem += choices[idx] == 0

        status = problem.solve()
        status_name = LpStatus.get(problem.status, None)
        if status_name != "Optimal":
            break

        selected = [i for i in range(n) if choices[i].value() == 1]
        if not selected:
            break

        excluded_indices.update(selected)
        solutions.append(selected)

    # Format output
    result_sets = []
    for sel in solutions:
        subset = food_df.iloc[sel][['name', 'calories', 'protein']].copy()
        subset = subset.reset_index(drop=True)
        total_calories = float(subset['calories'].sum())
        total_protein = float(subset['protein'].sum())
        items = subset.to_dict(orient='records')
        result_sets.append({
            "items": items,
            "total_calories": total_calories,
            "total_protein": total_protein,
            "indices": sel
        })

    return result_sets


@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    """
    Request JSON example:
    {
      "calories": 350.0,
      "weight_difference_percentage": 0,
      "current_weight": 70.0,            
      "num_sets": 3,                     
      "items_per_set": 3                 
    }
    """
    data = request.get_json(force=True)
    if 'calories' not in data or 'weight_difference_percentage' not in data:
        return jsonify({"error": "Provide 'calories' and 'weight_difference_percentage'"}), 400

    try:
        calories = float(data['calories'])
        weight_diff_pct = float(data['weight_difference_percentage'])
        num_sets = int(data.get('num_sets', 3))
        items_per_set = int(data.get('items_per_set', 3))
        current_weight = data.get('current_weight', None)
        max_rows = data.get('max_rows', None)
        if max_rows is not None:
            max_rows = int(max_rows)
    except Exception as e:
        return jsonify({"error": "Invalid input types", "detail": str(e)}), 400

    try:
        sets = recommend_foods(calories, weight_diff_pct, num_sets=num_sets,
                               items_per_set=items_per_set, max_rows=max_rows)
    except Exception as e:
        return jsonify({"error": "Recommendation error", "detail": str(e)}), 500

    if not sets:
        return jsonify({"error": "No feasible recommendations found for given calorie range."}), 404

    # If current_weight provided, compute estimated new weight per set
    for s in sets:
        if current_weight is not None:
            s['estimated_new_weight'] = calculate_new_weight(float(current_weight), float(s['total_calories']))

    return jsonify({"recommendation_sets": sets})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
