import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
from slope_stability.core.feature_utils import create_features, prepare_input_data
from slope_stability.core.model_utils import EnsemblePredictor

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'slope_stability', 'results', 'model_components_v2')

PREDICTION_THRESHOLD = 0.716

def get_model_dir_info():
    info = {
        'model_dir': MODEL_DIR,
        'model_dir_abs': os.path.abspath(MODEL_DIR),
        'exists': os.path.exists(MODEL_DIR),
        'is_dir': os.path.isdir(MODEL_DIR),
        'files': [],
        '__file__': __file__,
        'cwd': os.getcwd(),
    }
    if os.path.exists(MODEL_DIR) and os.path.isdir(MODEL_DIR):
        info['files'] = os.listdir(MODEL_DIR)
    return info

def load_model():
    try:
        model = EnsemblePredictor(MODEL_DIR)
        model.load()
        return model
    except FileNotFoundError as e:
        dir_info = get_model_dir_info()
        error_detail = f"""
**Model Loading Error: File Not Found**

**Model Directory:** {dir_info['model_dir_abs']}
**Directory Exists:** {'✅ Yes' if dir_info['exists'] else '❌ No'}
**Is Directory:** {'✅ Yes' if dir_info['is_dir'] else '❌ No'}

**Current Working Directory:** {dir_info['cwd']}
**Script Location:** {dir_info['__file__']}

**Files in Model Directory:**
{', '.join(dir_info['files']) if dir_info['files'] else '(empty)'}

**Error Details:** {str(e)}

**Possible Solutions:**
1. Ensure `slope_stability/results/model_components_v2/` directory exists with all model files
2. Check that all required files are present: scaler.pkl, weights.pkl, model_xgb.json, model_cat.cbm, model_lgb.pkl, model_rf.pkl, model_et.pkl, model_gb.pkl
3. Verify the directory structure is correct when deploying to Streamlit Cloud
4. If using Git, ensure model files are committed to the repository (consider Git LFS for large files)
        """
        st.error(error_detail, icon="🚨")
        return None
    except Exception as e:
        dir_info = get_model_dir_info()
        error_detail = f"""
**Model Loading Error: {type(e).__name__}**

**Model Directory:** {dir_info['model_dir_abs']}
**Directory Exists:** {'✅ Yes' if dir_info['exists'] else '❌ No'}

**Error Details:** {str(e)}

**Possible Solutions:**
1. Check that all required dependencies are installed
2. Verify model files are not corrupted
3. Ensure Python version is compatible with the model files
        """
        st.error(error_detail, icon="🚨")
        return None

def predict_slope_stability(model, unit_weight, cohesion, internal_friction_angle,
                            slope_angle, slope_height, pore_water_pressure_ratio):
    input_df = prepare_input_data(unit_weight, cohesion, internal_friction_angle,
                                  slope_angle, slope_height, pore_water_pressure_ratio)
    
    X_enhanced = create_features(input_df)
    
    stability_prob = model.predict_proba(X_enhanced)[0]
    
    is_stable = stability_prob >= PREDICTION_THRESHOLD
    
    return {
        'is_stable': bool(is_stable),
        'stability_probability': float(stability_prob),
        'threshold': PREDICTION_THRESHOLD
    }

def init_session_state():
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None

def add_prediction_record(result, inputs):
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'unit_weight': inputs['unit_weight'],
        'cohesion': inputs['cohesion'],
        'internal_friction_angle': inputs['internal_friction_angle'],
        'slope_angle': inputs['slope_angle'],
        'slope_height': inputs['slope_height'],
        'pore_water_pressure_ratio': inputs['pore_water_pressure_ratio'],
        'status': 'Stable' if result['is_stable'] else 'Unstable',
        'probability': result['stability_probability'] * 100,
        'threshold': result['threshold'] * 100
    }
    st.session_state.predictions.insert(0, record)
    if len(st.session_state.predictions) > 50:
        st.session_state.predictions = st.session_state.predictions[:50]

def clear_history():
    st.session_state.predictions = []

def main():
    st.set_page_config(
        page_title="Slope Stability Prediction",
        page_icon="⛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    model = load_model()
    if model is None:
        return

    with st.sidebar:
        st.header("Parameter Input")
        st.divider()

        unit_weight = st.number_input(
            "Unit Weight γ (kN/m³)",
            min_value=0.0, max_value=1000.0, value=210.0, step=10.0,
            help="Soil unit weight"
        )

        cohesion = st.number_input(
            "Cohesion C (kPa)",
            min_value=0.0, max_value=200.0, value=45.0, step=1.0,
            help="Soil cohesion"
        )

        internal_friction_angle = st.number_input(
            "Internal Friction Angle φ (°)",
            min_value=0.0, max_value=60.0, value=25.0, step=1.0,
            help="Angle of internal friction"
        )

        slope_angle = st.number_input(
            "Slope Angle β (°)",
            min_value=0.0, max_value=90.0, value=45.0, step=1.0,
            help="Slope inclination angle"
        )

        slope_height = st.number_input(
            "Slope Height H (m)",
            min_value=0.0, max_value=500.0, value=21.0, step=0.5,
            help="Height of the slope"
        )

        pore_water_pressure_ratio = st.number_input(
            "Pore Water Pressure Ratio r_u",
            min_value=0.0, max_value=1.0, value=0.2, step=0.01,
            help="Pore water pressure ratio"
        )

        st.divider()

        if st.button("Start Prediction", type="primary", use_container_width=True):
            inputs = {
                'unit_weight': unit_weight,
                'cohesion': cohesion,
                'internal_friction_angle': internal_friction_angle,
                'slope_angle': slope_angle,
                'slope_height': slope_height,
                'pore_water_pressure_ratio': pore_water_pressure_ratio
            }
            
            with st.spinner("Predicting..."):
                result = predict_slope_stability(model, **inputs)
                st.session_state.prediction_result = result
                add_prediction_record(result, inputs)

        st.divider()

        st.markdown("""
        **Parameter Description:**
        - γ: Unit Weight (kg/m³)
        - C: Cohesion (kPa)
        - φ: Internal Friction Angle (°)
        - β: Slope Angle (°)
        - H: Slope Height (m)
        - r_u: Pore Water Pressure Ratio
        """)

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            st.subheader("Prediction Result")
            
            if result['is_stable']:
                st.success("✅ Stable")
                color = "#2ecc71"
            else:
                st.error("❌ Unstable")
                color = "#e74c3c"
            
            st.divider()
            
            st.markdown(f"### Stability Probability")
            prob = result['stability_probability'] * 100
            st.progress(prob / 100, text=f"{prob:.2f}%")
            
            threshold = result['threshold'] * 100
            st.markdown(f"**Decision Threshold:** {threshold:.1f}%")

    with col2:
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            
            st.subheader("Prediction Details")
            
            status_color = "#27ae60" if result['is_stable'] else "#c0392b"
            
            st.markdown(f"""
            <div style="background-color: {status_color}; padding: 30px; border-radius: 16px; text-align: center;">
                <h2 style="color: white; font-size: 48px; margin-bottom: 10px;">
                    {'✅' if result['is_stable'] else '❌'}
                </h2>
                <h1 style="color: white; font-size: 36px;">
                    {'STABLE' if result['is_stable'] else 'UNSTABLE'}
                </h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("""
            **Prediction Logic:**
            - Probability ≥ 71.6% → Stable
            - Probability < 71.6% → Unstable
            """)
            
            st.markdown(f"""
            **Model Architecture:**
            - Optimized Ensemble Learning
            - Base Models: XGBoost, LightGBM, CatBoost, RF, ET, GB
            - Feature Engineering: 35+ derived features
            """)

    st.divider()
    
    st.subheader("Prediction History")
    
    if st.session_state.predictions:
        history_df = pd.DataFrame(st.session_state.predictions)
        
        history_df['status_color'] = history_df['status'].apply(
            lambda x: 'background-color: #d4edda' if x == 'Stable' else 'background-color: #f8d7da'
        )
        
        st.dataframe(
            history_df[[
                'timestamp', 'unit_weight', 'cohesion', 'internal_friction_angle',
                'slope_angle', 'slope_height', 'pore_water_pressure_ratio',
                'status', 'probability', 'threshold'
            ]].rename(columns={
                'unit_weight': 'γ (kg/m³)',
                'cohesion': 'C (kPa)',
                'internal_friction_angle': 'φ (°)',
                'slope_angle': 'β (°)',
                'slope_height': 'H (m)',
                'pore_water_pressure_ratio': 'r_u',
                'probability': 'Probability (%)',
                'threshold': 'Threshold (%)'
            }),
            hide_index=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                'Probability (%)': st.column_config.ProgressColumn(
                    format="%.2f%%", min_value=0, max_value=100
                ),
                'Threshold (%)': st.column_config.NumberColumn(format="%.0f%%")
            },
            use_container_width=True,
            height=400
        )
        
        if st.button("Clear History"):
            clear_history()
            st.rerun()
    else:
        st.info("No prediction history yet. Make a prediction to see results here.")

if __name__ == '__main__':
    main()
