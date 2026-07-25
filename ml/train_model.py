"""
Social Media Addiction Analyzer - ML Training Script
Stage 1: Predict Mental Health Score, Conflicts, Affects Academic from user inputs
Stage 2: Use all features to predict Addiction Score and Productivity Level
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Students_Social_Media_Addiction.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH).dropna()

    le_gender       = LabelEncoder()
    le_academic     = LabelEncoder()
    le_platform     = LabelEncoder()
    le_affects      = LabelEncoder()
    le_relationship = LabelEncoder()

    df['Gender_enc']       = le_gender.fit_transform(df['Gender'])
    df['Academic_enc']     = le_academic.fit_transform(df['Academic_Level'])
    df['Platform_enc']     = le_platform.fit_transform(df['Most_Used_Platform'])
    df['Affects_enc']      = le_affects.fit_transform(df['Affects_Academic_Performance'])
    df['Relationship_enc'] = le_relationship.fit_transform(df['Relationship_Status'])

    for le, name in [(le_gender,'le_gender'),(le_academic,'le_academic'),
                     (le_platform,'le_platform'),(le_affects,'le_affects'),
                     (le_relationship,'le_relationship')]:
        joblib.dump(le, os.path.join(MODEL_DIR, f'{name}.pkl'))

    joblib.dump({
        'gender':              list(le_gender.classes_),
        'academic_level':      list(le_academic.classes_),
        'platform':            list(le_platform.classes_),
        'affects_academic':    list(le_affects.classes_),
        'relationship_status': list(le_relationship.classes_)
    }, os.path.join(MODEL_DIR, 'encoder_classes.pkl'))

    # Productivity label
    df['Productivity_Score'] = (
        df['Sleep_Hours_Per_Night'] * 1.5 +
        df['Mental_Health_Score']   * 1.2 -
        df['Avg_Daily_Usage_Hours'] * 1.0 -
        df['Conflicts_Over_Social_Media'] * 0.5
    )
    pct = df['Productivity_Score'].quantile([0.33, 0.66])
    df['Productivity_Label'] = pd.cut(
        df['Productivity_Score'],
        bins=[-np.inf, pct[0.33], pct[0.66], np.inf],
        labels=['Low', 'Medium', 'High']
    )
    return df

# ── STAGE 1 features (what user directly provides) ──
STAGE1_FEATURES = [
    'Age', 'Gender_enc', 'Academic_enc',
    'Avg_Daily_Usage_Hours', 'Platform_enc',
    'Sleep_Hours_Per_Night', 'Relationship_enc'
]

# ── STAGE 2 features (stage1 + predicted intermediates) ──
STAGE2_FEATURES = STAGE1_FEATURES + [
    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Affects_enc'
]

def train_mental_health_model(df):
    X, y = df[STAGE1_FEATURES], df['Mental_Health_Score']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    print(f"Mental Health Model  → MAE: {mean_absolute_error(yte,pred):.3f}  R²: {r2_score(yte,pred):.3f}")
    joblib.dump(m, os.path.join(MODEL_DIR, 'mental_health_model.pkl'))
    return m

def train_conflicts_model(df):
    X, y = df[STAGE1_FEATURES], df['Conflicts_Over_Social_Media']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    print(f"Conflicts Model      → MAE: {mean_absolute_error(yte,pred):.3f}  R²: {r2_score(yte,pred):.3f}")
    joblib.dump(m, os.path.join(MODEL_DIR, 'conflicts_model.pkl'))
    return m

def train_affects_model(df):
    X, y = df[STAGE1_FEATURES], df['Affects_enc']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    m.fit(Xtr, ytr)
    print(f"Affects Academic     → Accuracy: {accuracy_score(yte, m.predict(Xte)):.3f}")
    joblib.dump(m, os.path.join(MODEL_DIR, 'affects_model.pkl'))
    return m

def train_addiction_model(df):
    X, y = df[STAGE2_FEATURES], df['Addicted_Score']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    cv = cross_val_score(m, X, y, cv=5, scoring='r2')
    print(f"Addiction Score      → MAE: {mean_absolute_error(yte,pred):.3f}  R²: {r2_score(yte,pred):.3f}  CV: {cv.mean():.3f}±{cv.std():.3f}")
    joblib.dump(m, os.path.join(MODEL_DIR, 'addiction_model.pkl'))
    return m

def train_productivity_model(df):
    valid = df.dropna(subset=['Productivity_Label'])
    X, y = valid[STAGE2_FEATURES], valid['Productivity_Label']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    m.fit(Xtr, ytr)
    print(f"Productivity Model   → Accuracy: {accuracy_score(yte, m.predict(Xte)):.3f}")
    joblib.dump(m, os.path.join(MODEL_DIR, 'productivity_model.pkl'))
    return m

def save_dataset_stats(df):
    stats = {
        'total_students':        int(len(df)),
        'avg_daily_usage':       round(df['Avg_Daily_Usage_Hours'].mean(), 2),
        'avg_addiction_score':   round(df['Addicted_Score'].mean(), 2),
        'avg_sleep':             round(df['Sleep_Hours_Per_Night'].mean(), 2),
        'avg_mental_health':     round(df['Mental_Health_Score'].mean(), 2),
        'affects_academic_pct':  round((df['Affects_Academic_Performance']=='Yes').mean()*100, 1),
        'platform_distribution': df['Most_Used_Platform'].value_counts().to_dict(),
        'gender_distribution':   df['Gender'].value_counts().to_dict(),
        'academic_distribution': df['Academic_Level'].value_counts().to_dict(),
        'addiction_by_platform': df.groupby('Most_Used_Platform')['Addicted_Score'].mean().round(2).to_dict(),
        'usage_by_academic':     df.groupby('Academic_Level')['Avg_Daily_Usage_Hours'].mean().round(2).to_dict(),
        'sleep_vs_score_corr':   round(df['Sleep_Hours_Per_Night'].corr(df['Addicted_Score']), 3),
        'usage_vs_score_corr':   round(df['Avg_Daily_Usage_Hours'].corr(df['Addicted_Score']), 3),
        'addiction_bins': {
            'Low (1-3)':      int((df['Addicted_Score'] <= 3).sum()),
            'Moderate (4-6)': int(((df['Addicted_Score'] > 3) & (df['Addicted_Score'] <= 6)).sum()),
            'High (7-9)':     int((df['Addicted_Score'] > 6).sum()),
        }
    }
    joblib.dump(stats, os.path.join(MODEL_DIR, 'dataset_stats.pkl'))
    print(f"Stats saved. Avg addiction: {stats['avg_addiction_score']}, Students: {stats['total_students']}")

def main():
    print("Loading data...")
    df = load_and_preprocess()
    print(f"Dataset: {len(df)} students\n")
    print("=== Stage 1: Intermediate Predictors ===")
    train_mental_health_model(df)
    train_conflicts_model(df)
    train_affects_model(df)
    print("\n=== Stage 2: Final Outputs ===")
    train_addiction_model(df)
    train_productivity_model(df)
    save_dataset_stats(df)
    print("\n✅ All models saved to ml/models/")

if __name__ == '__main__':
    main()
