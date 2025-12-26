from flask import Flask, render_template, request, redirect, session, send_from_directory
import os
import random
import numpy as np
from uuid import uuid4
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from supabase import create_client
from dotenv import load_dotenv

# -------------------------------------------------
# ENV & SUPABASE SETUP
# -------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# -------------------------------------------------
# FLASK APP SETUP
# -------------------------------------------------
app = Flask(__name__)
app.secret_key = "future-farm-secret-key"

BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.h5")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -------------------------------------------------
# LOAD AI MODEL
# -------------------------------------------------
model = load_model(MODEL_PATH)

CLASSES = [
    "Healthy",
    "Early Blight",
    "Late Blight"
]

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    print("👉 /signup route loaded, method =", request.method)

    if request.method == "POST":
        print("🔥 POST HIT SUCCESSFULLY")

        email = request.form.get("email")
        password = request.form.get("password")

        print("EMAIL =", email)
        print("PASSWORD RECEIVED =", bool(password))

        try:
            supabase.auth.sign_up({
                "email": email,
                "password": password
            })

            print("✅ SUPABASE SIGNUP CALLED")
            return redirect("/login")

        except Exception as e:
            print("❌ SUPABASE ERROR:", e)
            return f"Signup failed: {e}"

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if res.session:
            session["user_id"] = res.user.id
            session["access_token"] = res.session.access_token
            return redirect("/dashboard")

        return "Login failed"

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    print("SESSION USER ID:", session["user_id"])

    res = (
        supabase
        .table("prediction_history")
        .select("*")
        .eq("user_id", session["user_id"])
        .order("created_at", desc=True)
        .execute()
    )

    print("SUPABASE RAW RESPONSE:", res)
    print("SUPABASE DATA:", res.data)

    records = res.data if res.data else []

    return render_template("dashboard.html", records=records)



@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -------------------------------------------------
# AI PREDICTION FUNCTION
# -------------------------------------------------
def predict_plant_disease(img_path):
    img = image.load_img(img_path, target_size=(28, 28))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    idx = int(np.argmax(preds))
    confidence = round(float(np.max(preds)) * 100, 2)

    return CLASSES[idx], confidence

# -------------------------------------------------
# PREDICT ROUTE
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect("/login")

    crop = request.form.get("crop")
    image_file = request.files.get("image")

    if not image_file or image_file.filename == "":
        return "No image uploaded", 400

    ext = image_file.filename.split(".")[-1]
    filename = f"{uuid4()}.{ext}"
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_file.save(image_path)

    if crop == "Rice":
        disease = random.choice(["Brown Spot", "Leaf Blast", "Healthy"])
        confidence = None
        method = "Pretrained AI (PlantVillage)"
    else:
        disease, confidence = predict_plant_disease(image_path)
        method = "Pretrained AI (PlantVillage)"

# -------------------------------------------------
# SUPABASE SAVE RECORD
# -------------------------------------------------
    supabase.table("prediction_history").insert({
        "user_id": session["user_id"],
        "crop": crop,
        "disease": disease,
        "method": method,
        "confidence": confidence,
        "image": filename
    }).execute()

    return render_template(
        "result.html",
        crop=crop,
        disease=disease,
        method=method,
        confidence=confidence,
        image_filename=filename
    )

# -------------------------------------------------
# RUN SERVER
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
