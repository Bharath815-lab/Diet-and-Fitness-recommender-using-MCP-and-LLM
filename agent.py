# agent.py
import os
import sqlite3
import hashlib
import secrets
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash
from smolagents import ToolCallingAgent, ToolCollection, LiteLLMModel
from mcp import StdioServerParameters
from markdown import markdown
# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

# -------------------------
# Database Setup
# -------------------------
DB_PATH = os.getenv("DB_PATH", "fitness_users.db")

def init_db():
    """Initialize the SQLite database with users table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify a password against its hash"""
    return hash_password(password) == password_hash

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Initialize database on startup
init_db()

# -------------------------
# Authentication Decorator
# -------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# -------------------------
# LLM Configuration
# -------------------------
model = LiteLLMModel(
    model_id=os.getenv("LITELLM_MODEL_ID", "ollama_chat/gpt-oss:120b-cloud"),
    num_ctx=int(os.getenv("LITELLM_CTX", "8192")),
)

server_parameters = StdioServerParameters(
    command=os.getenv("PY_CMD", "python"),
    args=[os.getenv("SERVER_FILE", "server.py")],
)

# -------------------------
# Routes
# -------------------------

# Landing Page
@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")

# Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        if not username or not password:
            flash("Please fill in all fields.", "error")
            return render_template("login.html")
        
        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (username, username)
        ).fetchone()
        conn.close()
        
        if user and verify_password(password, user["password_hash"]):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("fitness_form"))
        else:
            flash("Invalid username or password.", "error")
    
    return render_template("login.html")

# Signup Page
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        # Validation
        if not all([username, email, password, confirm_password]):
            flash("Please fill in all fields.", "error")
            return render_template("signup.html")
        
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")
        
        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template("signup.html")
        
        conn = get_db_connection()
        try:
            password_hash = hash_password(password)
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            conn.commit()
            flash("Account created successfully! Please log in.", "success")
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
            conn.close()
    
    return render_template("signup.html")

# Logout
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("landing"))

# Fitness Form (Protected)
@app.route("/fitness-form", methods=["GET"])
@login_required
def fitness_form():
    return render_template("fitness-form.html")

# Form Submission (Protected)
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    form = request.form

    # Read user inputs
    gender = form["gender"]
    age = form["age"]
    height = form["height"]
    weight = form["weight"]
    duration = form["duration"]
    heart_rate = form["heart_rate"]
    body_temp = form["body_temp"]
    goal = form["goal"]

    # Prompt template
    prompt = f"""
You are a professional fitness assistant.

IMPORTANT:
- Format output using clean Markdown
- Use headings (##, ###)
- Use bullet points instead of long paragraphs
- Use tables for food plans
- Keep it structured and readable

User details:
- Gender: {gender}
- Age: {age}
- Height: {height} cm
- Weight: {weight} kg
- Workout Duration: {duration} minutes
- Heart Rate: {heart_rate}
- Body Temperature: {body_temp}

Goal:
{goal}

Output format:

## 🔥 Calories Burned
(Short answer)

## 🍽 Diet Plan
(Table format)

## 💪 Workout Plan
(Bullets)

## 📊 Macros
(Bullets)

## ✅ Tips
(Bullets)
"""

    # Run agent with MCP tools
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        agent = ToolCallingAgent(
            tools=tool_collection.tools,
            model=model
        )
        from markdown import markdown
        result = agent.run(prompt)
        html_result = markdown(result, extensions=["tables"])
        return render_template("result.html", result=html_result)

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=False)
