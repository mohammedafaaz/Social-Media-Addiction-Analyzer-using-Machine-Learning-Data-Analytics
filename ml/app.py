"""
Social Media Addiction Analyzer - Flask API (Two-Stage Pipeline)
Stage 1: Predict mental_health, conflicts, affects_academic from user inputs
Stage 2: Predict addiction_score and productivity using all features
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, os, pandas as pd

app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ── Feature names for proper model prediction ──
STAGE1_FEATURES = [
    'Age', 'Gender_enc', 'Academic_enc',
    'Avg_Daily_Usage_Hours', 'Platform_enc',
    'Sleep_Hours_Per_Night', 'Relationship_enc'
]

STAGE2_FEATURES = STAGE1_FEATURES + [
    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Affects_enc'
]

addiction_model    = joblib.load(os.path.join(MODEL_DIR, 'addiction_model.pkl'))
productivity_model = joblib.load(os.path.join(MODEL_DIR, 'productivity_model.pkl'))
mental_model       = joblib.load(os.path.join(MODEL_DIR, 'mental_health_model.pkl'))
conflicts_model    = joblib.load(os.path.join(MODEL_DIR, 'conflicts_model.pkl'))
affects_model      = joblib.load(os.path.join(MODEL_DIR, 'affects_model.pkl'))
encoder_classes    = joblib.load(os.path.join(MODEL_DIR, 'encoder_classes.pkl'))
dataset_stats      = joblib.load(os.path.join(MODEL_DIR, 'dataset_stats.pkl'))

le_gender       = joblib.load(os.path.join(MODEL_DIR, 'le_gender.pkl'))
le_academic     = joblib.load(os.path.join(MODEL_DIR, 'le_academic.pkl'))
le_platform     = joblib.load(os.path.join(MODEL_DIR, 'le_platform.pkl'))
le_affects      = joblib.load(os.path.join(MODEL_DIR, 'le_affects.pkl'))
le_relationship = joblib.load(os.path.join(MODEL_DIR, 'le_relationship.pkl'))

print("✅ All models loaded.")

def enc(encoder, value):
    try:    return int(encoder.transform([value])[0])
    except: return 0

def get_sleep_quality(hours):
    if hours >= 8:   return ('Excellent', '#22c55e', 95)
    if hours >= 7:   return ('Good',      '#86efac', 78)
    if hours >= 6:   return ('Fair',      '#f97316', 52)
    return               ('Poor',      '#ef4444', 28)

def get_mental_health_label(score):
    s = round(score)
    if s >= 8: return ('Thriving',   '#22c55e')
    if s >= 6: return ('Stable',     '#86efac')
    if s >= 4: return ('Struggling', '#f97316')
    return             ('At Risk',   '#ef4444')

def get_conflict_label(score):
    s = round(score)
    if s <= 1: return ('Minimal',  '#22c55e')
    if s <= 2: return ('Low',      '#86efac')
    if s <= 3: return ('Moderate', '#f97316')
    return             ('High',    '#ef4444')

def generate_recommendations(addiction, usage, sleep, mental, conflicts, affects, productivity):
    recs = []

    # Addiction risk
    if addiction >= 7:
        recs.append({'type':'critical','icon':'fa-circle-exclamation','title':'High Addiction Risk',
            'text':f'Your predicted addiction score of {addiction:.1f}/9 is in the danger zone. A structured digital detox plan is strongly recommended.'})
    elif addiction >= 5:
        recs.append({'type':'warning','icon':'fa-triangle-exclamation','title':'Moderate Addiction',
            'text':f'Your score of {addiction:.1f}/9 indicates growing dependency. Setting daily screen-time limits can significantly help.'})
    else:
        recs.append({'type':'success','icon':'fa-circle-check','title':'Healthy Digital Balance',
            'text':f'Your addiction score of {addiction:.1f}/9 reflects a healthy relationship with social media. Keep maintaining this balance.'})

    # Screen time
    if usage > 6:
        recs.append({'type':'critical','icon':'fa-circle-exclamation','title':'Excessive Screen Time',
            'text':f'{usage}h/day is significantly above healthy levels. Use app timers and schedule tech-free hours daily.'})
    elif usage > 4:
        recs.append({'type':'warning','icon':'fa-triangle-exclamation','title':'Above-Average Usage',
            'text':f'{usage}h/day is above the recommended limit. Try the 20-20-20 rule and reduce evening screen time.'})
    else:
        recs.append({'type':'success','icon':'fa-circle-check','title':'Controlled Usage',
            'text':f'{usage}h/day is within a manageable range. Continue being mindful of your screen habits.'})

    # Sleep
    if sleep < 6:
        recs.append({'type':'critical','icon':'fa-circle-exclamation','title':'Severe Sleep Deficit',
            'text':f'Only {sleep}h of sleep is harmful to both mental health and addiction recovery. Set a phone-free bedtime routine.'})
    elif sleep < 7:
        recs.append({'type':'warning','icon':'fa-triangle-exclamation','title':'Insufficient Sleep',
            'text':f'{sleep}h of sleep is below the recommended 7–9 hours. Avoid screens at least 1 hour before bed.'})
    else:
        recs.append({'type':'success','icon':'fa-circle-check','title':'Healthy Sleep Pattern',
            'text':f'{sleep}h of sleep is great. Good sleep is the single strongest buffer against addiction escalation.'})

    # Mental health
    if mental < 4:
        recs.append({'type':'critical','icon':'fa-circle-exclamation','title':'Mental Health Alert',
            'text':f'Your predicted mental health score of {mental:.1f}/10 is low. Social media overuse is likely contributing — consider speaking with a counselor.'})
    elif mental < 6:
        recs.append({'type':'warning','icon':'fa-triangle-exclamation','title':'Mental Health Check-In',
            'text':f'A mental health score of {mental:.1f}/10 has room to improve. Mindfulness practices and offline social time can help.'})
    else:
        recs.append({'type':'success','icon':'fa-circle-check','title':'Positive Mental Wellbeing',
            'text':f'Your predicted mental health score of {mental:.1f}/10 is healthy. Protect it by keeping your digital habits balanced.'})

    # Conflicts
    if conflicts >= 4:
        recs.append({'type':'critical','icon':'fa-circle-exclamation','title':'High Social Conflict',
            'text':f'A conflict score of {conflicts:.0f}/5 indicates that social media is damaging your relationships. Consider a temporary social media break.'})
    elif conflicts >= 2:
        recs.append({'type':'warning','icon':'fa-triangle-exclamation','title':'Some Social Friction',
            'text':f'Moderate conflict levels suggest some relationship strain from social media use. Prioritize face-to-face interactions.'})

    # Academic impact
    if affects == 'Yes':
        recs.append({'type':'warning','icon':'fa-triangle-exclamation','title':'Academic Impact Detected',
            'text':'Your usage pattern is likely affecting your academic performance. Try app blockers during study sessions and set a daily study-first rule.'})

    # Productivity
    if productivity == 'Low':
        recs.append({'type':'critical','icon':'fa-circle-exclamation','title':'Low Productivity Forecast',
            'text':'Your current habits point to low academic productivity. The Pomodoro technique and social media scheduling can create immediate improvement.'})
    elif productivity == 'Medium':
        recs.append({'type':'warning','icon':'fa-triangle-exclamation','title':'Moderate Productivity',
            'text':'You have moderate productivity potential. Small reductions in usage and better sleep could push you into the High tier.'})
    else:
        recs.append({'type':'success','icon':'fa-circle-check','title':'High Productivity',
            'text':'Your balanced habits support high academic productivity. You are on the right track — maintain consistency.'})

    return recs


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        d = request.get_json()

        age          = int(d.get('age', 20))
        gender       = d.get('gender', 'Male')
        academic     = d.get('academic_level', 'Undergraduate')
        usage        = float(d.get('avg_daily_usage_hours', 3))
        platform     = d.get('most_used_platform', 'Instagram')
        sleep        = float(d.get('sleep_hours_per_night', 7))
        relationship = d.get('relationship_status', 'Single')
        # Note: platforms_used field is optional and provided for future extension
        # Currently we use most_used_platform for backward compatibility
        _ = d.get('platforms_used', [])

        # Stage 1 vector
        s1_data = [[
            age, enc(le_gender, gender), enc(le_academic, academic),
            usage, enc(le_platform, platform), sleep, enc(le_relationship, relationship)
        ]]
        s1 = pd.DataFrame(s1_data, columns=STAGE1_FEATURES)

        mental_pred   = float(np.clip(mental_model.predict(s1)[0], 1, 10))
        conflicts_pred = float(np.clip(conflicts_model.predict(s1)[0], 0, 5))
        affects_enc_pred = int(affects_model.predict(s1)[0])
        affects_label = le_affects.inverse_transform([affects_enc_pred])[0]

        # Stage 2 vector
        s2_data = [[
            age, enc(le_gender, gender), enc(le_academic, academic),
            usage, enc(le_platform, platform), sleep, enc(le_relationship, relationship),
            mental_pred, conflicts_pred, affects_enc_pred
        ]]
        s2 = pd.DataFrame(s2_data, columns=STAGE2_FEATURES)

        addiction    = float(np.clip(addiction_model.predict(s2)[0], 1, 9))
        productivity = str(productivity_model.predict(s2)[0])
        prod_proba   = productivity_model.predict_proba(s2)[0]
        prod_classes = list(productivity_model.classes_)

        risk = 'High' if addiction >= 7 else ('Moderate' if addiction >= 5 else 'Low')
        risk_color = '#ef4444' if risk == 'High' else ('#f97316' if risk == 'Moderate' else '#22c55e')

        sleep_quality, sleep_color, sleep_pct = get_sleep_quality(sleep)
        mh_label, mh_color = get_mental_health_label(mental_pred)
        cf_label, cf_color = get_conflict_label(conflicts_pred)

        recs = generate_recommendations(addiction, usage, sleep, mental_pred,
                                         conflicts_pred, affects_label, productivity)

        return jsonify({
            'success': True,
            # Final outputs
            'addiction_score':     round(addiction, 2),
            'addiction_score_pct': round((addiction / 9) * 100, 1),
            'risk_level':          risk,
            'risk_color':          risk_color,
            'productivity_level':  productivity,
            'productivity_probabilities': dict(zip(prod_classes, [round(p*100,1) for p in prod_proba])),
            'score_vs_avg':        round(addiction - dataset_stats['avg_addiction_score'], 2),
            # Intermediate predictions
            'predicted_mental_health':  round(mental_pred, 1),
            'mental_health_label':      mh_label,
            'mental_health_color':      mh_color,
            'predicted_conflicts':      round(conflicts_pred, 1),
            'conflict_label':           cf_label,
            'conflict_color':           cf_color,
            'predicted_affects_academic': affects_label,
            # Sleep analysis
            'sleep_quality':  sleep_quality,
            'sleep_color':    sleep_color,
            'sleep_pct':      sleep_pct,
            # Extra derived
            'usage_vs_avg':   round(usage - dataset_stats['avg_daily_usage'], 2),
            'recommendations': recs
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/stats',   methods=['GET'])
def get_stats():
    return jsonify({'success': True, 'stats': dataset_stats})

@app.route('/api/options', methods=['GET'])
def get_options():
    return jsonify({'success': True, 'options': encoder_classes})

@app.route('/api/health',  methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("🚀 API running at http://localhost:5000")
    app.run(debug=True, port=5000)
