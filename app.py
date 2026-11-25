import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Page Config
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide")

# Custom CSS with Animated Ambient Background & Smooth Animations
st.markdown("""
<style>
    @keyframes bgPulse {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    @keyframes floatOrb {
        0%, 100% {
            transform: translate(0px, 0px) scale(1);
        }
        50% {
            transform: translate(35px, -25px) scale(1.08);
        }
    }

    @keyframes popIn {
        0% {
            opacity: 0;
            transform: translateY(14px) scale(0.98);
        }
        100% {
            opacity: 1;
            transform: translateY(0px) scale(1);
        }
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 12% 18%, rgba(99, 102, 241, 0.24), transparent 38%),
            radial-gradient(circle at 88% 22%, rgba(236, 72, 153, 0.20), transparent 40%),
            radial-gradient(circle at 50% 88%, rgba(6, 182, 212, 0.18), transparent 42%),
            linear-gradient(125deg, #070b14, #0f172a, #191638, #090d16);
        background-size: 200% 200%;
        animation: bgPulse 14s ease-in-out infinite;
        position: relative;
        overflow-x: hidden;
    }

    /* Subtle floating ambient accent layer */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: -120px;
        right: -120px;
        width: 360px;
        height: 360px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(245, 158, 11, 0.14), transparent 70%);
        filter: blur(24px);
        pointer-events: none;
        z-index: 0;
        animation: floatOrb 9s ease-in-out infinite;
    }

    .hero-box {
        background: linear-gradient(110deg, #6366f1 0%, #ec4899 52%, #f59e0b 100%);
        background-size: 180% 180%;
        animation: bgPulse 10s ease infinite, popIn 0.5s ease-out;
        padding: 24px 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 12px 28px rgba(99, 102, 241, 0.35);
    }
    .hero-box h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }
    .hero-box p {
        margin: 6px 0 0 0;
        font-size: 1rem;
        color: #f8fafc;
        opacity: 0.95;
    }
    .section-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(129, 140, 248, 0.45);
        color: #c7d2fe;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 12px;
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .section-pill:hover {
        transform: translateX(4px);
        background: rgba(99, 102, 241, 0.3);
    }
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 22px;
        backdrop-filter: blur(12px);
        animation: popIn 0.65s ease-out;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #06b6d4 0%, #6366f1 50%, #ec4899 100%);
        background-size: 180% 180%;
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 12px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background-position 0.4s ease;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) scale(1.01);
        background-position: 100% 50%;
        box-shadow: 0 10px 24px rgba(236, 72, 153, 0.42);
        color: white;
    }
    .result-card {
        padding: 22px;
        border-radius: 18px;
        color: white;
        margin-top: 16px;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.4);
        animation: popIn 0.45s cubic-bezier(0.22, 1, 0.36, 1);
    }
</style>
""", unsafe_allow_html=True)

# Hero Banner
st.markdown("""
<div class="hero-box">
    <h1>📱 Mobile Price Range Classification</h1>
    <p>Specs set karein aur ML model se phone ki estimated price range turant dekhein.</p>
</div>
""", unsafe_allow_html=True)

# Load & Train Model
@st.cache_resource
def load_model():
    data = pd.read_csv("mobile_price_range_data.csv")
    X = data.drop("price_range", axis=1)
    y = data["price_range"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist()

model, scaler, feature_cols = load_model()

# Input Form
with st.form("specs_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-pill">⚡ Hardware & Performance</div>', unsafe_allow_html=True)
        battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1200)
        ram = st.slider("RAM (MB)", 256, 4096, 2048)
        int_memory = st.slider("Internal Memory (GB)", 2, 64, 32)
        clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
        n_cores = st.selectbox("Processor Cores", [1, 2, 3, 4, 5, 6, 7, 8], index=3)
        pc = st.slider("Primary Camera (MP)", 0, 20, 12)
        fc = st.slider("Front Camera (MP)", 0, 19, 5)
        mobile_wt = st.slider("Mobile Weight (g)", 80, 200, 140)
        m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.1)
        talk_time = st.slider("Talk Time (Hours)", 2, 20, 10)

    with col2:
        st.markdown('<div class="section-pill">🖥️ Display & Connectivity</div>', unsafe_allow_html=True)
        px_height = st.slider("Pixel Resolution Height", 0, 1960, 650)
        px_width = st.slider("Pixel Resolution Width", 500, 1998, 1250)
        sc_h = st.slider("Screen Height (cm)", 5, 19, 12)
        sc_w = st.slider("Screen Width (cm)", 0, 18, 6)

        st.markdown("**Connectivity & Features**")
        c_a, c_b = st.columns(2)
        with c_a:
            four_g = st.selectbox("4G Support", [0, 1], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No", index=1)
            wifi = st.selectbox("Wi-Fi", [0, 1], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No", index=1)
            dual_sim = st.selectbox("Dual SIM", [0, 1], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No", index=1)
        with c_b:
            three_g = st.selectbox("3G Support", [0, 1], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No", index=1)
            blue = st.selectbox("Bluetooth", [0, 1], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No", index=1)
            touch_screen = st.selectbox("Touch Screen", [0, 1], format_func=lambda x: "✅ Yes" if x == 1 else "❌ No", index=1)

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("✨ Predict Price Range", use_container_width=True)

if submit:
    input_dict = {
        "battery_power": battery_power,
        "blue": blue,
        "clock_speed": clock_speed,
        "dual_sim": dual_sim,
        "fc": fc,
        "four_g": four_g,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": three_g,
        "touch_screen": touch_screen,
        "wifi": wifi,
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]
    input_scaled = scaler.transform(input_df)
    prediction = int(model.predict(input_scaled)[0])

    price_themes = {
        0: ("Low Cost (Budget Segment)", "₹5,000 – ₹10,000 (Approx.)", "linear-gradient(135deg, #059669, #10b981)", "💚"),
        1: ("Medium Cost (Mid-Range)", "₹10,000 – ₹20,000 (Approx.)", "linear-gradient(135deg, #0284c7, #38bdf8)", "💙"),
        2: ("High Cost (Upper Mid-Range)", "₹20,000 – ₹35,000 (Approx.)", "linear-gradient(135deg, #d97706, #fbbf24)", "🧡"),
        3: ("Very High Cost (Flagship)", "₹35,000+ (Approx.)", "linear-gradient(135deg, #db2777, #9333ea)", "💜"),
    }

    label, approx_price, bg_gradient, badge = price_themes[prediction]

    st.markdown(f"""
    <div class="result-card" style="background: {bg_gradient};">
        <div style="font-size: 0.95rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;">
            {badge} Prediction Result • Class {prediction}
        </div>
        <div style="font-size: 1.75rem; font-weight: 800; margin: 6px 0;">
            {label}
        </div>
        <div style="font-size: 1.1rem; font-weight: 600;">
            Estimated Price Bracket: {approx_price}
        </div>
    </div>
    """, unsafe_allow_html=True)