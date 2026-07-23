
# ==============================================================================
# FILE 1: requirements.txt (Required by Vercel)
# Create a separate file named requirements.txt in your repository and paste this:
# 
# Flask==3.0.2
# Werkzeug==3.0.1
# scikit-learn==1.6.1
# numpy==1.26.4
# pandas==2.2.0
# scipy==1.12.0
# ==============================================================================

# ==============================================================================
# FILE 2: vercel.json (Required by Vercel)
# Create a separate file named vercel.json in your repository and paste this:
# 
# {
#   "version": 2,
#   "builds": [{"src": "app.py", "use": "@vercel/python"}],
#   "routes": [{"src": "/(.*)", "dest": "app.py"}]
# }
# ==============================================================================

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- MODEL LOADING & ARCHITECTURE INFERENCE ---
# The model is a DecisionTreeClassifier[cite: 2]
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'DecisionTree.pkl')

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
        
    # Extract feature names from the Decision Tree[cite: 2]
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    else:
        # Fallback to the exact features identified in the pickle dump[cite: 2]
        feature_names = ['Age', 'Gender', 'Region', 'Occupation', 'Income']
        
    # Decision Trees use feature_importances_ (0 to 1 magnitude), unlike Logistic Regression
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_.tolist()
    else:
        feature_importance = [1.0 / len(feature_names)] * len(feature_names)
        
except Exception as e:
    print("Error loading Decision Tree model: " + str(e))
    model = None
    feature_names = ['Age', 'Gender', 'Region', 'Occupation', 'Income']
    feature_importance = [0.2, 0.2, 0.2, 0.2, 0.2]

# --- ADVANCED HTML / CSS / JS DASHBOARD TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Futuristic Decision Tree Analytics</title>
    <!-- Chart.js for High-Performance Canvas Rendering -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* =========================================
           THEME DEFINITIONS & CSS VARIABLES
           ========================================= */
        :root {
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --transition-speed: 0.4s;
            --glass-blur: 16px;
            --shadow-opacity: 0.2;
        }

        /* 1. Default: Midnight Glass */
        body.theme-default {
            --bg-color: #0b1120;
            --surface-glass: rgba(30, 41, 59, 0.6);
            --surface-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #3b82f6; 
            --accent-secondary: #14b8a6; 
            --danger: #ef4444;
            --success: #10b981;
            --bg-gradient: radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.15) 0%, transparent 40%),
                           radial-gradient(circle at 85% 85%, rgba(20, 184, 166, 0.15) 0%, transparent 40%);
        }

        /* 2. Cyberpunk: High Contrast Neon */
        body.theme-cyberpunk {
            --bg-color: #050510;
            --surface-glass: rgba(15, 5, 25, 0.7);
            --surface-border: rgba(255, 0, 124, 0.25);
            --text-primary: #ffffff;
            --text-secondary: #f472b6;
            --accent-primary: #ff007c; 
            --accent-secondary: #00f0ff; 
            --danger: #ff1111;
            --success: #00ff9d;
            --bg-gradient: radial-gradient(circle at 10% 20%, rgba(255, 0, 124, 0.2) 0%, transparent 45%),
                           radial-gradient(circle at 90% 80%, rgba(0, 240, 255, 0.15) 0%, transparent 45%);
        }

        /* 3. Deep Forest: Organic & Earthy */
        body.theme-forest {
            --bg-color: #051008;
            --surface-glass: rgba(10, 25, 15, 0.75);
            --surface-border: rgba(46, 160, 67, 0.2);
            --text-primary: #e6f4ea;
            --text-secondary: #8ab4f8;
            --accent-primary: #2ea043; 
            --accent-secondary: #a8b545; 
            --danger: #ff5252;
            --success: #4caf50;
            --bg-gradient: radial-gradient(circle at 20% 30%, rgba(46, 160, 67, 0.15) 0%, transparent 50%),
                           radial-gradient(circle at 80% 70%, rgba(168, 181, 69, 0.1) 0%, transparent 50%);
        }

        /* 4. Glacial Ice: Clean & Crisp */
        body.theme-ice {
            --bg-color: #000c18;
            --surface-glass: rgba(5, 20, 40, 0.65);
            --surface-border: rgba(56, 189, 248, 0.25);
            --text-primary: #f0f9ff;
            --text-secondary: #7dd3fc;
            --accent-primary: #38bdf8; 
            --accent-secondary: #818cf8; 
            --danger: #fb7185;
            --success: #2dd4bf;
            --bg-gradient: radial-gradient(circle at 0% 0%, rgba(56, 189, 248, 0.2) 0%, transparent 60%),
                           radial-gradient(circle at 100% 100%, rgba(129, 140, 248, 0.15) 0%, transparent 60%);
        }

        /* 5. Magic Purple: Mystical */
        body.theme-magic {
            --bg-color: #0c0014;
            --surface-glass: rgba(25, 5, 45, 0.7);
            --surface-border: rgba(168, 85, 247, 0.3);
            --text-primary: #faf5ff;
            --text-secondary: #d8b4fe;
            --accent-primary: #a855f7; 
            --accent-secondary: #ec4899; 
            --danger: #ef4444;
            --success: #d946ef;
            --bg-gradient: radial-gradient(circle at 30% 20%, rgba(168, 85, 247, 0.2) 0%, transparent 50%),
                           radial-gradient(circle at 70% 80%, rgba(236, 72, 153, 0.15) 0%, transparent 50%);
        }

        /* 6. Solar Flare: High Energy */
        body.theme-solar {
            --bg-color: #1a0500;
            --surface-glass: rgba(40, 10, 0, 0.8);
            --surface-border: rgba(249, 115, 22, 0.3);
            --text-primary: #fff7ed;
            --text-secondary: #fdba74;
            --accent-primary: #f97316; 
            --accent-secondary: #facc15; 
            --danger: #ef4444;
            --success: #84cc16;
            --bg-gradient: radial-gradient(circle at 50% 0%, rgba(249, 115, 22, 0.2) 0%, transparent 60%);
        }

        /* 7. Abyssal Void: Minimalist Dark */
        body.theme-void {
            --bg-color: #000000;
            --surface-glass: rgba(10, 10, 10, 0.9);
            --surface-border: rgba(255, 255, 255, 0.1);
            --text-primary: #ffffff;
            --text-secondary: #a3a3a3;
            --accent-primary: #ffffff; 
            --accent-secondary: #525252; 
            --danger: #7f1d1d;
            --success: #14532d;
            --bg-gradient: none;
        }

        /* 8. Neon Synthwave: Retro 80s */
        body.theme-synthwave {
            --bg-color: #0f0518;
            --surface-glass: rgba(20, 10, 40, 0.7);
            --surface-border: rgba(249, 115, 22, 0.4);
            --text-primary: #ffffff;
            --text-secondary: #c4b5fd;
            --accent-primary: #f97316; 
            --accent-secondary: #a855f7; 
            --danger: #e11d48;
            --success: #2dd4bf;
            --bg-gradient: linear-gradient(180deg, rgba(15, 5, 24, 1) 0%, rgba(40, 10, 60, 0.5) 100%);
        }

        /* =========================================
           GLOBAL LAYOUT & TYPOGRAPHY
           ========================================= */
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: var(--font-family);
            background-color: var(--bg-color);
            background-image: var(--bg-gradient);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            transition: background-color var(--transition-speed) ease, background-image var(--transition-speed) ease;
        }

        .container {
            width: 98%;
            max-width: 1600px;
            margin: 0 auto 3rem auto;
            display: grid;
            grid-template-columns: 380px 1fr;
            gap: 2rem;
            padding: 0 1rem;
        }

        /* =========================================
           UI COMPONENTS: GLASSMORPHISM CARDS
           ========================================= */
        .glass-card {
            background: var(--surface-glass);
            backdrop-filter: blur(var(--glass-blur));
            -webkit-backdrop-filter: blur(var(--glass-blur));
            border: 1px solid var(--surface-border);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, var(--shadow-opacity));
            transition: transform var(--transition-speed) cubic-bezier(0.4, 0, 0.2, 1), 
                        box-shadow var(--transition-speed) ease, 
                        background var(--transition-speed) ease, 
                        border-color var(--transition-speed) ease;
            position: relative;
            overflow: hidden;
        }
        
        /* Inner glow effect for futuristic look */
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
            opacity: 0.3;
        }

        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
            border-color: var(--accent-primary);
        }

        /* =========================================
           HEADER & NAVIGATION
           ========================================= */
        .top-nav {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            padding: 1.5rem 2.5rem 0;
            z-index: 10;
        }

        .theme-selector {
            appearance: none;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--surface-border);
            color: var(--text-primary);
            padding: 0.75rem 2.5rem 0.75rem 1.5rem;
            border-radius: 12px;
            font-family: var(--font-family);
            font-weight: 500;
            letter-spacing: 0.5px;
            cursor: pointer;
            transition: all 0.3s ease;
            background-image: url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%2394a3b8%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E");
            background-repeat: no-repeat, repeat;
            background-position: right .7em top 50%, 0 0;
            background-size: .65em auto, 100%;
        }
        .theme-selector:focus {
            outline: none;
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }
        .theme-selector option { background: #111; color: #fff; }

        .app-header {
            text-align: center;
            padding: 2rem 1rem 3rem 1rem;
            animation: fadeInDown 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .app-header h1 {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -1.5px;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-primary) 50%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            transition: all var(--transition-speed) ease;
        }

        /* =========================================
           FORM INPUTS & SIDEBAR
           ========================================= */
        .form-container {
            max-height: calc(100vh - 200px);
            overflow-y: auto;
            padding-right: 15px;
        }
        
        /* Custom Scrollbar */
        .form-container::-webkit-scrollbar { width: 8px; }
        .form-container::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 4px; }
        .form-container::-webkit-scrollbar-thumb { background: var(--surface-border); border-radius: 4px; }
        .form-container::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

        .input-group { margin-bottom: 1.5rem; position: relative; }
        .input-group label {
            display: block;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            transition: color var(--transition-speed) ease;
        }
        
        .input-group input {
            width: 100%;
            padding: 1rem 1.2rem;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .input-group input:focus {
            outline: none;
            border-color: var(--accent-primary);
            background: rgba(0, 0, 0, 0.5);
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
            transform: translateY(-2px);
        }

        .btn-predict {
            width: 100%;
            padding: 1.2rem;
            margin-top: 1rem;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-transform: uppercase;
            letter-spacing: 2px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .btn-predict::after {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: linear-gradient(transparent, rgba(255,255,255,0.2), transparent);
            transform: rotate(45deg) translateY(-100%);
            transition: all 0.5s ease;
        }
        .btn-predict:hover::after { transform: rotate(45deg) translateY(100%); }
        .btn-predict:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.5); }
        .btn-predict:active { transform: translateY(1px); }

        /* =========================================
           DASHBOARD GRID & ANALYTICS PANELS
           ========================================= */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            align-items: stretch;
        }
        
        .full-width { grid-column: 1 / -1; }
        
        h3.panel-title {
            font-size: 1.1rem;
            color: var(--text-secondary);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid var(--surface-border);
            padding-bottom: 0.75rem;
        }

        /* Result Display Area */
        .result-display {
            text-align: center;
            padding: 3rem 2rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            position: relative;
        }
        
        .result-status {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 1.5rem 0;
            transition: color var(--transition-speed) ease, text-shadow 0.3s ease;
            letter-spacing: -0.5px;
        }
        
        /* Dynamic Logo Styling */
        .logo-container svg {
            width: 72px;
            height: 72px;
            stroke: currentColor;
            stroke-width: 1.5;
            transition: all var(--transition-speed) ease;
            filter: drop-shadow(0 0 10px currentColor);
        }
        
        /* KPI Metrics Cards */
        .metric-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            width: 100%;
            margin-top: 1rem;
        }
        
        .metric {
            text-align: center;
            padding: 1.5rem 1rem;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 16px;
            border: 1px solid var(--surface-border);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric:hover {
            border-color: var(--accent-primary);
            transform: translateY(-5px);
            background: rgba(0, 0, 0, 0.6);
        }
        
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--accent-secondary);
            transition: color var(--transition-speed) ease;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        /* Chart Containers */
        .chart-container { position: relative; height: 320px; width: 100%; }
        .radar-container { position: relative; height: 350px; width: 100%; }

        /* =========================================
           ANIMATIONS & KEYFRAMES
           ========================================= */
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulseAlert {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 20px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        
        .animate-up { animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; opacity: 0; }
        .delay-1 { animation-delay: 0.1s; }
        .delay-2 { animation-delay: 0.2s; }
        .delay-3 { animation-delay: 0.3s; }
        .delay-4 { animation-delay: 0.4s; }

        /* High Risk Alert Class applied dynamically via JS */
        .alert-pulse { animation: pulseAlert 2s infinite; border-color: var(--danger) !important; }

        /* =========================================
           RESPONSIVE DESIGN
           ========================================= */
        @media (max-width: 1200px) {
            .container { grid-template-columns: 320px 1fr; }
        }
        @media (max-width: 968px) {
            .container { grid-template-columns: 1fr; }
            .dashboard-grid { grid-template-columns: 1fr; }
            .metric-cards { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
            .form-container { max-height: none; overflow: visible; }
        }
    </style>
</head>
<body class="theme-default">

    <!-- THEME SWITCHER -->
    <nav class="top-nav animate-up">
        <select id="themeSelector" class="theme-selector" onchange="changeTheme(this.value)">
            <option value="theme-default">Theme: Midnight Glass</option>
            <option value="theme-cyberpunk">Theme: Cyberpunk Core</option>
            <option value="theme-forest">Theme: Deep Forest</option>
            <option value="theme-ice">Theme: Glacial Ice</option>
            <option value="theme-magic">Theme: Magic Amethyst</option>
            <option value="theme-solar">Theme: Solar Flare</option>
            <option value="theme-void">Theme: Abyssal Void</option>
            <option value="theme-synthwave">Theme: Neon Synthwave</option>
        </select>
    </nav>

    <!-- HEADER -->
    <header class="app-header">
        <h1>Predictive Analytics Engine</h1>
        <p style="font-size: 1.2rem; color: var(--text-secondary); font-weight: 300; letter-spacing: 1px;">
            Advanced Decision Tree Node Classifier
        </p>
    </header>

    <div class="container">
        
        <!-- =========================================
             SIDEBAR: USER INPUT (DECISION TREE FEATURES)
             ========================================= -->
        <aside class="glass-card animate-up delay-1">
            <h3 class="panel-title">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
                Student Profile Input
            </h3>
            
            <div class="form-container">
                <form id="predictionForm">
                    <!-- Script to generate exact features found in DecisionTree.pkl[cite: 2] -->
                    <script>
                        const dtFeatures = [
                            {id: 'Age', label: 'Age', type: 'number', step: '1', val: 21},
                            {id: 'Gender', label: 'Gender (0=M, 1=F)', type: 'number', step: '1', val: 1},
                            {id: 'Region', label: 'Region Code', type: 'number', step: '1', val: 3},
                            {id: 'Occupation', label: 'Occupation Code', type: 'number', step: '1', val: 2},
                            {id: 'Income', label: 'Household Income ($)', type: 'number', step: '100', val: 35000}
                        ];
                        
                        // Strict string concatenation to prevent Python template rendering errors
                        dtFeatures.forEach(f => {
                            document.write(
                                '<div class="input-group">' +
                                '<label for="' + f.id + '">' + f.label + '</label>' +
                                '<input type="' + f.type + '" id="' + f.id + '" name="' + f.id + '" step="' + f.step + '" value="' + f.val + '" required>' +
                                '</div>'
                            );
                        });
                    </script>
                    
                    <button type="submit" class="btn-predict" id="submitBtn">Execute Neural Inference</button>
                </form>
            </div>
        </aside>

        <!-- =========================================
             MAIN DASHBOARD: ADVANCED ANALYTICS
             ========================================= -->
        <main class="dashboard-grid">
            
            <!-- 1. PRIMARY RESULT PANEL -->
            <div class="glass-card full-width result-display animate-up delay-2" id="resultCard">
                <h3 class="panel-title" style="position: absolute; top: 1.5rem; left: 2rem; border:none;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    Inference Engine Status
                </h3>
                
                <div id="resultOutput" class="result-status">
                    <span style="font-size:1.2rem; color: var(--text-secondary); font-weight: 400;">Awaiting Matrix Input...</span>
                </div>
                
                <div class="metric-cards">
                    <div class="metric">
                        <div id="probStay" class="metric-value">--%</div>
                        <div class="metric-label">Retention Confidence</div>
                    </div>
                    <div class="metric">
                        <div id="probDrop" class="metric-value">--%</div>
                        <div class="metric-label">Attrition Probability</div>
                    </div>
                    <div class="metric">
                        <div id="riskLevel" class="metric-value" style="color: #fff;">--</div>
                        <div class="metric-label">Assessed Risk Factor</div>
                    </div>
                </div>
            </div>

            <!-- 2. DONUT CHART (PROBABILITY) -->
            <div class="glass-card animate-up delay-3">
                <h3 class="panel-title">Confidence Distribution</h3>
                <div class="chart-container"><canvas id="donutChart"></canvas></div>
            </div>

            <!-- 3. POLAR AREA CHART (FEATURE DISTRIBUTION) -->
            <div class="glass-card animate-up delay-3">
                <h3 class="panel-title">Feature Vector Analysis</h3>
                <div class="chart-container"><canvas id="polarChart"></canvas></div>
            </div>

            <!-- 4. RADAR CHART (PROFILE) -->
            <div class="glass-card animate-up delay-4">
                <h3 class="panel-title">Behavioral Topology</h3>
                <div class="radar-container"><canvas id="radarChart"></canvas></div>
            </div>

            <!-- 5. BAR CHART (DECISION TREE IMPORTANCE) -->
            <div class="glass-card animate-up delay-4">
                <h3 class="panel-title">Decision Tree Node Importance Mapping</h3>
                <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem; line-height: 1.4;">
                    Visualizes the absolute Gini importance extracted directly from the DecisionTreeClassifier model architecture.
                </p>
                <div class="chart-container" style="height: 280px;"><canvas id="barChart"></canvas></div>
            </div>
            
        </main>
    </div>

    <!-- =========================================
         JAVASCRIPT: LOGIC & CHART RENDERING
         ========================================= -->
    <script>
        // --- DATA SETUP FROM PYTHON BACKEND ---
        const globalFeatures = {{ features | tojson | safe }};
        const globalImportances = {{ importance | tojson | safe }};
        
        // Globals for Chart Instances
        let donutChart, radarChart, barChart, polarChart;

        // --- SVG LOGOS (String Concatenation for safe Python rendering) ---
        const SVG_DROPOUT = '<div class="logo-container"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/></svg></div>';
        
        const SVG_RETENTION = '<div class="logo-container"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg></div>';

        // --- THEME COLOR ENGINE (Expanded) ---
        const themePalette = {
            'theme-default':   { primary: '#3b82f6', secondary: '#14b8a6', danger: '#ef4444', success: '#10b981', grid: 'rgba(255,255,255,0.05)', text: '#94a3b8' },
            'theme-cyberpunk': { primary: '#ff007c', secondary: '#00f0ff', danger: '#ff1111', success: '#00ff9d', grid: 'rgba(255,0,124,0.15)',  text: '#f472b6' },
            'theme-forest':    { primary: '#2ea043', secondary: '#a8b545', danger: '#ff5252', success: '#4caf50', grid: 'rgba(46,160,67,0.1)',   text: '#8ab4f8' },
            'theme-ice':       { primary: '#38bdf8', secondary: '#818cf8', danger: '#fb7185', success: '#2dd4bf', grid: 'rgba(56,189,248,0.1)',  text: '#7dd3fc' },
            'theme-magic':     { primary: '#a855f7', secondary: '#ec4899', danger: '#ef4444', success: '#d946ef', grid: 'rgba(168,85,247,0.1)',  text: '#d8b4fe' },
            'theme-solar':     { primary: '#f97316', secondary: '#facc15', danger: '#ef4444', success: '#84cc16', grid: 'rgba(249,115,22,0.1)',  text: '#fdba74' },
            'theme-void':      { primary: '#ffffff', secondary: '#525252', danger: '#7f1d1d', success: '#14532d', grid: 'rgba(255,255,255,0.05)', text: '#a3a3a3' },
            'theme-synthwave': { primary: '#f97316', secondary: '#a855f7', danger: '#e11d48', success: '#2dd4bf', grid: 'rgba(168,85,247,0.15)', text: '#c4b5fd' }
        };

        // Utility to generate RGBA colors dynamically
        function hexToRgba(hex, alpha) {
            let r = parseInt(hex.slice(1, 3), 16),
                g = parseInt(hex.slice(3, 5), 16),
                b = parseInt(hex.slice(5, 7), 16);
            return 'rgba(' + r + ', ' + g + ', ' + b + ', ' + alpha + ')';
        }

        // Generate an array of gradient colors for the Polar chart based on current theme
        function getPaletteArray(theme) {
            return [
                hexToRgba(theme.primary, 0.7),
                hexToRgba(theme.secondary, 0.7),
                hexToRgba(theme.success, 0.7),
                hexToRgba(theme.danger, 0.7),
                hexToRgba('#a855f7', 0.7) // Neutral accent
            ];
        }

        // --- GLOBAL CHART.JS CONFIGURATION ---
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.85)';
        Chart.defaults.plugins.tooltip.titleColor = '#fff';
        Chart.defaults.plugins.tooltip.padding = 12;
        Chart.defaults.plugins.tooltip.cornerRadius = 8;
        Chart.defaults.plugins.legend.labels.usePointStyle = true;

        // --- INITIALIZE ALL CHARTS ---
        function initCharts() {
            const currentTheme = themePalette['theme-default'];
            Chart.defaults.color = currentTheme.text;

            // 1. DONUT CHART
            const ctxDonut = document.getElementById('donutChart').getContext('2d');
            donutChart = new Chart(ctxDonut, {
                type: 'doughnut',
                data: { 
                    labels: ['Retention (0)', 'Dropout (1)'], 
                    datasets: [{ 
                        data: [50, 50], 
                        backgroundColor: [currentTheme.success, currentTheme.danger], 
                        borderWidth: 0,
                        hoverOffset: 10
                    }] 
                },
                options: { 
                    responsive: true, maintainAspectRatio: false, cutout: '75%', 
                    plugins: { legend: { position: 'bottom' } }, 
                    animation: { animateScale: true, animateRotate: true, duration: 1500 } 
                }
            });

            // 2. POLAR AREA CHART (New Analytics Feature)
            const ctxPolar = document.getElementById('polarChart').getContext('2d');
            polarChart = new Chart(ctxPolar, {
                type: 'polarArea',
                data: {
                    labels: globalFeatures,
                    datasets: [{
                        label: 'Input Topology',
                        data: [20, 20, 20, 20, 20], // Default dummy data
                        backgroundColor: getPaletteArray(currentTheme),
                        borderWidth: 1,
                        borderColor: 'rgba(0,0,0,0.5)'
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    scales: { r: { ticks: { display: false }, grid: { color: currentTheme.grid } } },
                    plugins: { legend: { position: 'right', labels: { boxWidth: 12 } } }
                }
            });

            // 3. RADAR CHART
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            radarChart = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: globalFeatures,
                    datasets: [{ 
                        label: 'Normalized Profile', 
                        data: [0,0,0,0,0], 
                        backgroundColor: hexToRgba(currentTheme.primary, 0.2), 
                        borderColor: currentTheme.primary, 
                        pointBackgroundColor: currentTheme.secondary, 
                        borderWidth: 2,
                        pointRadius: 4,
                        pointHoverRadius: 6
                    }]
                },
                options: { 
                    responsive: true, maintainAspectRatio: false, 
                    scales: { 
                        r: { 
                            angleLines: { color: currentTheme.grid }, 
                            grid: { color: currentTheme.grid }, 
                            pointLabels: { font: { size: 11 } }, 
                            ticks: { display: false } 
                        } 
                    } 
                }
            });

            // 4. BAR CHART (Decision Tree Importances)
            // Note: Importances are always positive in Decision Trees
            const bgColors = globalImportances.map(val => hexToRgba(currentTheme.primary, 0.6 + (val * 0.4)));
            const ctxBar = document.getElementById('barChart').getContext('2d');
            barChart = new Chart(ctxBar, {
                type: 'bar',
                data: { 
                    labels: globalFeatures, 
                    datasets: [{ 
                        label: 'Gini Importance', 
                        data: globalImportances, 
                        backgroundColor: bgColors, 
                        borderColor: currentTheme.primary, 
                        borderWidth: 1, 
                        borderRadius: 6 
                    }] 
                },
                options: { 
                    responsive: true, maintainAspectRatio: false, 
                    scales: { 
                        y: { grid: { color: currentTheme.grid }, beginAtZero: true }, 
                        x: { grid: { display: false } } 
                    }, 
                    plugins: { legend: { display: false } } 
                }
            });
        }

        // --- DYNAMIC THEME SWITCHER LOGIC ---
        function changeTheme(themeClass) {
            document.body.className = themeClass;
            const colors = themePalette[themeClass];
            
            // Update Global Fonts
            Chart.defaults.color = colors.text;

            // Update Donut
            donutChart.data.datasets[0].backgroundColor = [colors.success, colors.danger];
            donutChart.update();

            // Update Polar Area
            polarChart.data.datasets[0].backgroundColor = getPaletteArray(colors);
            polarChart.options.scales.r.grid.color = colors.grid;
            polarChart.update();

            // Update Radar
            radarChart.data.datasets[0].backgroundColor = hexToRgba(colors.primary, 0.2);
            radarChart.data.datasets[0].borderColor = colors.primary;
            radarChart.data.datasets[0].pointBackgroundColor = colors.secondary;
            radarChart.options.scales.r.angleLines.color = colors.grid;
            radarChart.options.scales.r.grid.color = colors.grid;
            radarChart.update();

            // Update Bar Chart
            const newBgColors = globalImportances.map(val => hexToRgba(colors.primary, 0.6 + (val * 0.4)));
            barChart.data.datasets[0].backgroundColor = newBgColors;
            barChart.data.datasets[0].borderColor = colors.primary;
            barChart.options.scales.y.grid.color = colors.grid;
            barChart.update();
            
            // Force redraw risk text and logo colors if already processed
            const riskSpan = document.getElementById('riskLevel');
            const resOut = document.getElementById('resultOutput');
            const resCard = document.getElementById('resultCard');
            
            if(riskSpan.textContent === 'Critical' || riskSpan.textContent === 'High') { 
                riskSpan.style.color = colors.danger; 
                resOut.style.color = colors.danger;
                resCard.style.borderColor = colors.danger;
            }
            else if(riskSpan.textContent === 'Low' || riskSpan.textContent === 'Minimal') { 
                riskSpan.style.color = colors.success; 
                resOut.style.color = colors.success;
                resCard.style.borderColor = 'transparent';
            }
        }

        // --- FORM SUBMISSION & API FETCH ---
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // UX: Button Loading State
            const btn = document.getElementById('submitBtn');
            const originalText = btn.innerHTML;
            btn.innerHTML = 'Processing Matrix...';
            btn.style.opacity = '0.8';
            btn.style.pointerEvents = 'none';

            const formData = new FormData(e.target);
            const dataObj = {};
            formData.forEach((value, key) => { dataObj[key] = parseFloat(value); });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(dataObj)
                });
                
                const result = await response.json();
                
                if(response.ok) { 
                    updateDashboard(result, dataObj); 
                } else { 
                    alert('Neural Engine Error: ' + result.error); 
                }
            } catch (error) { 
                alert('Connection terminated: ' + error.message); 
            } finally {
                // Restore Button
                btn.innerHTML = originalText;
                btn.style.opacity = '1';
                btn.style.pointerEvents = 'auto';
            }
        });

        // --- DASHBOARD UPDATER ---
        function updateDashboard(apiData, inputData) {
            const currentTheme = document.body.className;
            const colors = themePalette[currentTheme];

            // 1. UI Text & Logo Injection (Requested feature: output "student will dropout or not")
            const resOut = document.getElementById('resultOutput');
            const resCard = document.getElementById('resultCard');
            
            if (apiData.prediction_code === 1) {
                resOut.innerHTML = SVG_DROPOUT + '<div style="margin-top:10px; font-size:2rem; letter-spacing: -1px;">' + apiData.prediction + '</div>';
                resOut.style.color = colors.danger;
                resOut.style.textShadow = '0 0 20px ' + hexToRgba(colors.danger, 0.5);
                resCard.classList.add('alert-pulse');
            } else {
                resOut.innerHTML = SVG_RETENTION + '<div style="margin-top:10px; font-size:2rem; letter-spacing: -1px;">' + apiData.prediction + '</div>';
                resOut.style.color = colors.success;
                resOut.style.textShadow = '0 0 20px ' + hexToRgba(colors.success, 0.5);
                resCard.classList.remove('alert-pulse');
                resCard.style.borderColor = 'transparent';
                resCard.style.boxShadow = ''; // reset pulse
            }
            
            // 2. KPI Metrics
            // Formatting with counters for futuristic feel
            document.getElementById('probStay').textContent = apiData.probability_stay.toFixed(1) + '%';
            document.getElementById('probDrop').textContent = apiData.probability_dropout.toFixed(1) + '%';
            
            const riskSpan = document.getElementById('riskLevel');
            riskSpan.textContent = apiData.risk_level;
            if (apiData.prediction_code === 1) {
                riskSpan.style.color = colors.danger;
            } else {
                riskSpan.style.color = colors.success;
            }

            // 3. Update Donut Chart Data
            donutChart.data.datasets[0].data = [apiData.probability_stay, apiData.probability_dropout];
            donutChart.update();

            // 4. Update Polar Area Chart Data
            // We scale the raw inputs down to a relative visual distribution
            const rawInputs = [
                inputData.Age || 0,
                (inputData.Gender || 0) * 20, // Scale small numbers for visibility
                (inputData.Region || 0) * 10,
                (inputData.Occupation || 0) * 10,
                (inputData.Income || 0) / 1000 // Scale large numbers down
            ];
            polarChart.data.datasets[0].data = rawInputs;
            polarChart.update();

            // 5. Update Radar Chart (Normalized Profile)
            radarChart.data.datasets[0].data = [
                (inputData.Age / 50) * 100, // Assuming max age 50 for scaling
                inputData.Gender * 100,
                (inputData.Region / 10) * 100,
                (inputData.Occupation / 10) * 100,
                (inputData.Income / 100000) * 100 // Assuming max 100k for scaling
            ];
            radarChart.update();
        }

        // --- INITIALIZATION TRIGGER ---
        window.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>
"""

# ==============================================================================
# FLASK ROUTING & API ENDPOINTS
# ==============================================================================

@app.route('/')
def home():
    """Renders the main glassmorphism dashboard."""
    return render_template_string(
        HTML_TEMPLATE, 
        features=feature_names,
        importance=feature_importance
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the model inference for the Decision Tree via JSON payload."""
    if not model:
        return jsonify({'error': 'Decision Tree Model failed to load into memory.'}), 500

    try:
        # Extract payload from frontend
        data = request.get_json()
        
        # Build the exact feature array the model expects[cite: 2]
        input_data = []
        for feature in feature_names:
            val = float(data.get(feature, 0.0))
            input_data.append(val)
            
        # Reshape for Sklearn prediction (1 sample, n features)
        features_array = np.array(input_data).reshape(1, -1)
        
        # Execute Decision Tree methods
        prediction_val = int(model.predict(features_array)[0])
        
        # Handle probability logic (Decision trees sometimes output exact 0.0/1.0 probabilities at leaf nodes)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
        else:
            # Fallback if probability isn't supported
            probabilities = [1.0, 0.0] if prediction_val == 0 else [0.0, 1.0]
            
        # Format exact output string requested by user
        if prediction_val == 1:
            result_text = "Student will dropout"
            # Calculate dynamic risk phrasing
            risk = "Critical" if probabilities[1] > 0.8 else "High"
        else:
            result_text = "Student will not dropout"
            risk = "Minimal" if probabilities[0] > 0.8 else "Low"
            
        return jsonify({
            'prediction': result_text,
            'prediction_code': prediction_val,
            'probability_dropout': float(probabilities[1] * 100),
            'probability_stay': float(probabilities[0] * 100),
            'risk_level': risk,
            'input_echo': input_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Entry point for local testing
if __name__ == '__main__':
    app.run(debug=True, port=5000)
