import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CBC Teacher Readiness Diagnostic",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 5px solid #2a5298;
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .section-header {
        color: #1e3c72;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #28a745, #20c997);
    }
    
    .score-badge {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3c72;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .band-low {
        background: #dc3545;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: 600;
    }
    
    .band-moderate {
        background: #ffc107;
        color: #212529;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: 600;
    }
    
    .band-high {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: 600;
    }
    
    .info-tooltip {
        background: #e7f3ff;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
        color: #004085;
        border-left: 3px solid #004085;
        margin: 0.5rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0
if 'total_score' not in st.session_state:
    st.session_state.total_score = 0
if 'previous_predictions' not in st.session_state:
    st.session_state.previous_predictions = []

# Header Section
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">🎓 CBC Teacher Readiness Diagnostic</h1>
    <p style="margin:0; opacity:0.9">Evidence-based support for Competency-Based Curriculum implementation</p>
    <p style="margin:0; font-size:0.9rem; margin-top:0.5rem">Version 2.0 | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - About & Instructions
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/teacher.png", width=80)
    st.markdown("### About This Tool")
    st.markdown("""
    This diagnostic tool helps school administrators and education officers:
    
    - **Assess** individual teacher readiness for CBC
    - **Identify** specific areas needing support
    - **Simulate** intervention impact
    - **Generate** evidence-based recommendations
    """)
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Diagnostics Run", st.session_state.predictions_made)
    with col2:
        if st.session_state.predictions_made > 0:
            avg_score = st.session_state.total_score / st.session_state.predictions_made
            st.metric("Avg Score", f"{avg_score:.1f}")
    
    st.markdown("---")
    st.markdown("### Export Options")
    
    if st.button(" Export Session Data", use_container_width=True):
        if st.session_state.previous_predictions:
            export_df = pd.DataFrame(st.session_state.previous_predictions)
            csv = export_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="teacher_readiness_data.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("###  Need Help?")
    st.info("""
    For technical support or questions about interpretation, contact:
    **research@strathmore.edu**
    """)

# Load model with better error handling
@st.cache_resource
def load_assets():
    """Load the trained stacking ensemble model"""
    try:
        model = joblib.load('07_final_stacking_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure '07_final_stacking_model.pkl' is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

try:
    final_model = load_assets()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Individual Assessment", 
    " Intervention Simulator", 
    " Cohort Analysis",
    " Progress Tracking",
    " Technical Documentation"
])

# ============================================
# TAB 1: INDIVIDUAL ASSESSMENT (FIXED VERSION)
# ============================================
with tab1:
    st.markdown("##  Individual Teacher Readiness Assessment")
    st.markdown("Complete all four sections to generate a personalized readiness profile.")
    
    # Progress tracking
    sections_completed = 0
    
    # Create two columns for the main content
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # Section 1: Pedagogical Confidence (Psychological Pillar)
        with st.expander(" Section 1: Pedagogical Confidence", expanded=True):
            st.markdown("*How confident are you in these areas?*")
            
            col1, col2 = st.columns(2)
            with col1:
                # FIXED: Use options 1-5 only, no 0
                conf_B2 = st.selectbox(
                    "Classroom Management", 
                    [1, 2, 3, 4, 5], 
                    index=2,
                    format_func=lambda x: ["Select"][0] if x == 0 else ["Very Low", "Low", "Moderate", "High", "Very High"][x-1],
                    help="Ability to manage a learner-centered classroom effectively",
                    key="conf_B2_tab1"
                )
            with col2:
                conf_B6 = st.selectbox(
                    "Learner Engagement", 
                    [1, 2, 3, 4, 5], 
                    index=2,
                    format_func=lambda x: ["Select"][0] if x == 0 else ["Very Low", "Low", "Moderate", "High", "Very High"][x-1],
                    help="Confidence in engaging students in active learning",
                    key="conf_B6_tab1"
                )
            
            if conf_B2 and conf_B6:
                sections_completed += 1
        
        # Section 2: Digital Literacy
        with st.expander(" Section 2: Digital Literacy", expanded=True):
            st.markdown("*Rate your digital teaching capabilities*")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                # FIXED: Use options 1-5 only
                digi_C1c = st.selectbox(
                    "Content Creation", 
                    [1, 2, 3, 4, 5], 
                    index=2,
                    format_func=lambda x: ["Select"][0] if x == 0 else ["Very Low", "Low", "Moderate", "High", "Very High"][x-1],
                    help="Creating digital content for lessons",
                    key="digi_C1c_tab1"
                )
            with col2:
                digi_C1d = st.selectbox(
                    "Digital Assessment", 
                    [1, 2, 3, 4, 5], 
                    index=2,
                    format_func=lambda x: ["Select"][0] if x == 0 else ["Very Low", "Low", "Moderate", "High", "Very High"][x-1],
                    help="Using digital tools for assessment",
                    key="digi_C1d_tab1"
                )
            with col3:
                digi_C2e = st.selectbox(
                    "Digital Collaboration", 
                    [1, 2, 3, 4, 5], 
                    index=2,
                    format_func=lambda x: ["Select"][0] if x == 0 else ["Very Low", "Low", "Moderate", "High", "Very High"][x-1],
                    help="Facilitating online collaboration",
                    key="digi_C2e_tab1"
                )
            
            if digi_C1c and digi_C1d and digi_C2e:
                sections_completed += 1
        
        # Section 3: Resource Availability
        with st.expander(" Section 3: Resource Availability", expanded=True):
            st.markdown("*Assess your school's resources*")
            
            col1, col2 = st.columns(2)
            with col1:
                # FIXED: Use radio with 1-5 only (radio is fine as is)
                res_D1 = st.radio(
                    "ICT Infrastructure", 
                    [1, 2, 3, 4, 5], 
                    index=2, 
                    horizontal=True,
                    format_func=lambda x: ["Very Inadequate", "Inadequate", "Adequate", "Good", "Excellent"][x-1],
                    key="res_D1_tab1"
                )
                res_D2 = st.radio(
                    "Internet Connectivity", 
                    [1, 2, 3, 4, 5], 
                    index=2, 
                    horizontal=True,
                    format_func=lambda x: ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                    key="res_D2_tab1"
                )
                res_D3 = st.radio(
                    "Learning Materials", 
                    [1, 2, 3, 4, 5], 
                    index=2, 
                    horizontal=True,
                    format_func=lambda x: ["Very Inadequate", "Inadequate", "Adequate", "Good", "Excellent"][x-1],
                    key="res_D3_tab1"
                )
            with col2:
                res_D4 = st.radio(
                    "Technical Support", 
                    [1, 2, 3, 4, 5], 
                    index=2, 
                    horizontal=True,
                    format_func=lambda x: ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                    key="res_D4_tab1"
                )
                res_D5 = st.radio(
                    "Electricity Reliability", 
                    [1, 2, 3, 4, 5], 
                    index=2, 
                    horizontal=True,
                    format_func=lambda x: ["Very Unreliable", "Unreliable", "Fair", "Reliable", "Very Reliable"][x-1],
                    key="res_D5_tab1"
                )
                res_D6 = st.radio(
                    "Equipment Maintenance", 
                    [1, 2, 3, 4, 5], 
                    index=2, 
                    horizontal=True,
                    format_func=lambda x: ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                    key="res_D6_tab1"
                )
            
            sections_completed += 1
        
        # Section 4: Training Quality
        with st.expander(" Section 4: Training Quality", expanded=True):
            st.markdown("*Evaluate your CBC training experience*")
            
            train_attended = st.checkbox("I have attended CBC training", value=True, key="train_attended_tab1")
            
            if train_attended:
                col1, col2 = st.columns(2)
                with col1:
                    # FIXED: Use selectbox with 1-5 only
                    train_E2 = st.selectbox(
                        "Training Comprehensiveness", 
                        [1, 2, 3, 4, 5], 
                        index=2,
                        format_func=lambda x: ["Select"][0] if x == 0 else ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                        key="train_E2_tab1"
                    )
                    train_E3 = st.selectbox(
                        "Practical Applicability", 
                        [1, 2, 3, 4, 5], 
                        index=2,
                        format_func=lambda x: ["Select"][0] if x == 0 else ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                        key="train_E3_tab1"
                    )
                    train_E4 = st.selectbox(
                        "Trainer Effectiveness", 
                        [1, 2, 3, 4, 5], 
                        index=2,
                        format_func=lambda x: ["Select"][0] if x == 0 else ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                        key="train_E4_tab1"
                    )
                with col2:
                    train_E5 = st.selectbox(
                        "Practice Opportunities", 
                        [1, 2, 3, 4, 5], 
                        index=2,
                        format_func=lambda x: ["Select"][0] if x == 0 else ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                        key="train_E5_tab1"
                    )
                    train_E6 = st.selectbox(
                        "Follow-up Support", 
                        [1, 2, 3, 4, 5], 
                        index=2,
                        format_func=lambda x: ["Select"][0] if x == 0 else ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                        key="train_E6_tab1"
                    )
                    train_E7 = st.selectbox(
                        "Confidence Improvement", 
                        [1, 2, 3, 4, 5], 
                        index=2,
                        format_func=lambda x: ["Select"][0] if x == 0 else ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                        key="train_E7_tab1"
                    )
                training_attended_binary = 1.0
                
                # Check if all training items have values
                if all([train_E2, train_E3, train_E4, train_E5, train_E6, train_E7]):
                    sections_completed += 1
            else:
                # If no training, set all training quality to 1 (Very Poor)
                train_E2 = train_E3 = train_E4 = train_E5 = train_E6 = train_E7 = 1
                training_attended_binary = 0.0
                sections_completed += 1
                st.info(" Since no training was attended, training quality defaults to 'Very Poor'.")
            
            # Overall training quality slider (works fine as is)
            train_q = st.select_slider(
                "Overall Training Quality", 
                options=[1, 2, 3, 4, 5], 
                value=3,
                format_func=lambda x: ["Very Poor", "Poor", "Fair", "Good", "Excellent"][x-1],
                key="train_q_tab1"
            )
        
        # Action buttons
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            generate_btn = st.button(" Generate Readiness Profile", use_container_width=True, type="primary", key="generate_tab1")
    
    with right_col:
        # Display progress
        st.markdown("###  Assessment Progress")
        progress = sections_completed / 4
        st.progress(progress)
        st.markdown(f"**Completed:** {sections_completed}/4 sections")
        
        if progress == 1.0:
            st.success("✅ All sections complete! Ready for analysis.")
        else:
            remaining = 4 - sections_completed
            st.warning(f"⚠️ Please complete {remaining} more section(s)")
        
        st.markdown("---")
        
        # Display quick tips
        st.markdown("###  Quick Tips")
        st.info("""
        - Answer honestly for accurate results
        - All fields are required
        - Results are confidential
        - Use the simulator to test interventions
        """)
    
    # Results section (appears after generation)
    if generate_btn and sections_completed == 4:
        # Prepare input data
        Digital_x_Training = float(digi_C1c * ((train_E2 + train_E3 + train_E4 + train_E5 + train_E6 + train_E7)/6))
        
        data_dict = {
            'train_E4': float(train_E4), 'train_E3': float(train_E3), 'train_E7': float(train_E7),
            'train_E6': float(train_E6), 'train_E2': float(train_E2), 'train_E5': float(train_E5),
            'Digital_x_Training': Digital_x_Training, 'res_D6': float(res_D6),
            'training_attended_binary': training_attended_binary, 'res_D5': float(res_D5),
            'res_D4': float(res_D4), 'digi_C1d': float(digi_C1d), 'res_D3': float(res_D3),
            'res_D1': float(res_D1), 'res_D2': float(res_D2), 'digi_C1c': float(digi_C1c),
            'digi_C2e': float(digi_C2e), 'conf_B2': float(conf_B2), 'conf_B6': float(conf_B6)
        }
        
        expected_order = ['train_E4', 'train_E3', 'train_E7', 'train_E6', 'train_E2', 'train_E5',
                         'Digital_x_Training', 'res_D6', 'training_attended_binary', 'res_D5',
                         'res_D4', 'digi_C1d', 'res_D3', 'res_D1', 'res_D2', 'digi_C1c',
                         'digi_C2e', 'conf_B2', 'conf_B6']
        
        input_df = pd.DataFrame([data_dict])[expected_order]
        
        try:
            # Make prediction
            prediction = final_model.predict(input_df)[0]
            prediction = np.clip(prediction, 4.0, 20.0)  # TRI scale is 4-20
            
            # Update session stats
            st.session_state.predictions_made += 1
            st.session_state.total_score += prediction
            st.session_state.previous_predictions.append({
                'timestamp': pd.Timestamp.now(),
                'score': prediction,
                'conf_B2': conf_B2,
                'conf_B6': conf_B6,
                'digi_C1c': digi_C1c
            })
            
            # Display results in a nice layout
            st.markdown("---")
            st.markdown("##  Readiness Assessment Results")
            
            # Three columns for key metrics
            metric1, metric2, metric3 = st.columns(3)
            
            with metric1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("###  TRI Score")
                st.markdown(f'<div class="score-badge">{prediction:.1f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("###  Readiness Band")
                
                if prediction < 10:
                    band_class = "band-low"
                    band_text = "Low Readiness"
                elif prediction < 15:
                    band_class = "band-moderate"
                    band_text = "Moderate Readiness"
                else:
                    band_class = "band-high"
                    band_text = "High Readiness"
                
                st.markdown(f'<div class="{band_class}">{band_text}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("###  Percentile")
                # Approximate percentile based on sample distribution
                if prediction < 8:
                    percentile = 5
                elif prediction < 10:
                    percentile = 25
                elif prediction < 12:
                    percentile = 50
                elif prediction < 15:
                    percentile = 75
                else:
                    percentile = 95
                st.markdown(f"<h2 style='text-align:center'>~{percentile}th</h2>", unsafe_allow_html=True)
                st.caption("Compared to Nakuru County teachers")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Construct-level breakdown
            st.markdown("###  Construct-Level Breakdown")
            
            # Calculate construct means
            ped_mean = (conf_B2 + conf_B6) / 2
            dig_mean = (digi_C1c + digi_C1d + digi_C2e) / 3
            res_mean = (res_D1 + res_D2 + res_D3 + res_D4 + res_D5 + res_D6) / 6
            train_mean = (train_E2 + train_E3 + train_E4 + train_E5 + train_E6 + train_E7) / 6
            
            # Create radar chart
            categories = ['Pedagogical\nConfidence', 'Digital\nLiteracy', 'Resource\nAvailability', 'Training\nQuality']
            values = [ped_mean, dig_mean, res_mean, train_mean]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current Profile',
                line_color='#2a5298',
                fillcolor='rgba(42, 82, 152, 0.3)'
            ))
            
            # Add reference line for moderate readiness (3.0)
            fig.add_trace(go.Scatterpolar(
                r=[3, 3, 3, 3],
                theta=categories,
                fill='none',
                name='Moderate Threshold',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[1, 5]
                    )),
                showlegend=True,
                title="Readiness Profile by Construct",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate recommendations
            st.markdown("###  Personalized Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown("####  Priority Interventions")
                
                # Sort constructs by deficit
                deficits = [
                    ('Pedagogical Confidence', ped_mean, [
                        "Peer observation and coaching",
                        "Structured reflection exercises",
                        "Mastery experience opportunities"
                    ]),
                    ('Digital Literacy', dig_mean, [
                        "Hands-on tool workshops",
                        "Digital content creation training",
                        "Peer mentoring in tech integration"
                    ]),
                    ('Resource Availability', res_mean, [
                        "School-level resource advocacy",
                        "Community partnership building",
                        "Alternative resource identification"
                    ]),
                    ('Training Quality', train_mean, [
                        "Quality-focused PD selection",
                        "Follow-up support engagement",
                        "Peer learning community participation"
                    ])
                ]
                
                deficits.sort(key=lambda x: x[1])
                
                for construct, score, recs in deficits[:2]:  # Top 2 priority areas
                    st.markdown(f"**{construct}** (Score: {score:.1f}/5)")
                    for rec in recs[:2]:  # Top 2 recommendations
                        st.markdown(f"- {rec}")
                    st.markdown("---")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                st.markdown("####  Growth Opportunities")
                
                # Show strengths
                strengths = [(construct, score) for construct, score, _ in deficits if score >= 3.5]
                if strengths:
                    st.markdown("**Strengths to leverage:**")
                    for construct, score in strengths:
                        st.markdown(f"- ✅ {construct} ({score:.1f}/5)")
                else:
                    st.markdown("*Focus on building foundational skills first*")
                
                st.markdown("---")
                st.markdown("**Next Steps:**")
                st.markdown("""
                1. Share results with supervisor
                2. Develop action plan
                3. Schedule follow-up assessment in 3 months
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Threshold warning for Digital Literacy
            if train_mean < 3.0 and dig_mean > 3.5:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("""
                ### ⚠️ Important Finding
                **Your digital skills may not translate into readiness** because your training quality is below 3.0. 
                
                **Recommendation:** Prioritize foundational CBC training before intensive digital upskilling. Digital skills are most effective when paired with quality training.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.exception(e)

# ============================================
# TAB 2: INTERVENTION SIMULATOR (Keep as is)
# ============================================
with tab2:
    st.markdown("##  Intervention Impact Simulator")
    st.markdown("Model how different interventions could improve readiness scores.")
    
    st.info("""
    **How to use:** Adjust the sliders below to simulate improvements in each area.
    The simulator will show the predicted impact on overall readiness.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Current Profile")
        st.markdown("*(Use values from your last assessment)*")
        
        # Input fields for current scores
        current_ped = st.slider("Current Pedagogical Confidence", 1.0, 5.0, 2.5, 0.5, key="sim_ped")
        current_dig = st.slider("Current Digital Literacy", 1.0, 5.0, 2.5, 0.5, key="sim_dig")
        current_res = st.slider("Current Resource Availability", 1.0, 5.0, 2.5, 0.5, key="sim_res")
        current_train = st.slider("Current Training Quality", 1.0, 5.0, 2.5, 0.5, key="sim_train")
    
    with col2:
        st.markdown("### Target Profile")
        st.markdown("*(Set targets for intervention)*")
        
        target_ped = st.slider("Target Pedagogical Confidence", 1.0, 5.0, 3.5, 0.5, key="target_ped")
        target_dig = st.slider("Target Digital Literacy", 1.0, 5.0, 3.5, 0.5, key="target_dig")
        target_res = st.slider("Target Resource Availability", 1.0, 5.0, 3.5, 0.5, key="target_res")
        target_train = st.slider("Target Training Quality", 1.0, 5.0, 3.5, 0.5, key="target_train")
    
    if st.button("Simulate Impact", type="primary", key="simulate_btn"):
        # Current TRI
        current_tri = (current_ped + current_dig + current_res + current_train) * 4 / 4
        
        # Target TRI
        target_tri = (target_ped + target_dig + target_res + target_train) * 4 / 4
        
        # Calculate gains
        ped_gain = (target_ped - current_ped) * 1.0  # Weight based on model coefficients
        dig_gain = (target_dig - current_dig) * 0.7  # Approximate weight
        res_gain = (target_res - current_res) * 0.5
        train_gain = (target_train - current_train) * 0.5
        
        # Show results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current TRI", f"{current_tri:.1f}")
        with col2:
            st.metric("Target TRI", f"{target_tri:.1f}")
        with col3:
            st.metric("Potential Gain", f"+{target_tri - current_tri:.1f}", 
                     delta_color="normal")
        
        # Create gain chart
        fig = go.Figure(data=[
            go.Bar(name='Current', x=['Pedagogical', 'Digital', 'Resources', 'Training'], 
                   y=[current_ped, current_dig, current_res, current_train]),
            go.Bar(name='Target', x=['Pedagogical', 'Digital', 'Resources', 'Training'], 
                   y=[target_ped, target_dig, target_res, target_train])
        ])
        
        fig.update_layout(
            title="Intervention Impact by Construct",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Check threshold condition
        if current_train < 3.0 and target_dig > current_dig:
            st.warning("""
            ⚠️ **Note:** Digital skills improvements may have limited impact if training quality remains below 3.0.
            Consider prioritizing training quality first.
            """)

# ============================================
# TAB 3: COHORT ANALYSIS (MORE FLEXIBLE VERSION)
# ============================================
with tab3:
    st.markdown("##  Cohort Analysis")
    st.markdown("Upload multiple teacher records to analyze school-level or county-level readiness.")
    
    # Template download
    with st.expander(" Download Template"):
        template_df = pd.DataFrame({
            'conf_B2': [4, 3, 2],
            'conf_B6': [4, 3, 2],
            'digi_C1c': [3, 4, 2],
            'digi_C1d': [3, 3, 2],
            'digi_C2e': [4, 3, 2],
            'res_D1': [3, 4, 2],
            'res_D2': [3, 3, 2],
            'res_D3': [4, 3, 2],
            'res_D4': [3, 3, 2],
            'res_D5': [3, 4, 2],
            'res_D6': [3, 3, 2],
            'train_E2': [3, 4, 2],
            'train_E3': [3, 3, 2],
            'train_E4': [4, 3, 2],
            'train_E5': [3, 4, 2],
            'train_E6': [3, 3, 2],
            'train_E7': [4, 3, 2],
            'training_attended': [1, 1, 0]
        })
        
        csv = template_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="cohort_template.csv">📎 Click here to download template CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.caption("Use this template format for best results")
    
    uploaded_file = st.file_uploader("Upload CSV with teacher responses", type=['csv'])
    
    if uploaded_file is not None:
        try:
            cohort_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(cohort_df)} teacher records")
            
            # Display column names for debugging (remove this after it works)
            with st.expander(" Column Names in Your File"):
                st.write(cohort_df.columns.tolist())
            
            # Display preview
            st.markdown("### Data Preview")
            st.dataframe(cohort_df.head())
            
            # Define the columns we need with flexible matching
            required_patterns = {
                'ped': ['conf_B2', 'conf_B6'],
                'dig': ['digi_C1c', 'digi_C1d', 'digi_C2e'],
                'res': ['res_D1', 'res_D2', 'res_D3', 'res_D4', 'res_D5', 'res_D6'],
                'train': ['train_E2', 'train_E3', 'train_E4', 'train_E5', 'train_E6', 'train_E7']
            }
            
            # Find matching columns (case-insensitive)
            available_cols = {col.lower(): col for col in cohort_df.columns}
            
            found_cols = {'ped': [], 'dig': [], 'res': [], 'train': []}
            
            for category, patterns in required_patterns.items():
                for pattern in patterns:
                    pattern_lower = pattern.lower()
                    if pattern_lower in available_cols:
                        found_cols[category].append(available_cols[pattern_lower])
            
            # Calculate TRI if we have at least some columns
            if any(len(cols) > 0 for cols in found_cols.values()):
                
                # Calculate construct means
                if found_cols['ped']:
                    cohort_df['Pedagogical_Confidence'] = cohort_df[found_cols['ped']].mean(axis=1)
                else:
                    cohort_df['Pedagogical_Confidence'] = 2.5
                    st.warning("⚠️ Pedagogical Confidence columns not found, using default value 2.5")
                
                if found_cols['dig']:
                    cohort_df['Digital_Literacy'] = cohort_df[found_cols['dig']].mean(axis=1)
                else:
                    cohort_df['Digital_Literacy'] = 2.5
                    st.warning("⚠️ Digital Literacy columns not found, using default value 2.5")
                
                if found_cols['res']:
                    cohort_df['Resource_Availability'] = cohort_df[found_cols['res']].mean(axis=1)
                else:
                    cohort_df['Resource_Availability'] = 2.5
                    st.warning("⚠️ Resource Availability columns not found, using default value 2.5")
                
                if found_cols['train']:
                    cohort_df['Training_Quality'] = cohort_df[found_cols['train']].mean(axis=1)
                else:
                    cohort_df['Training_Quality'] = 2.5
                    st.warning("⚠️ Training Quality columns not found, using default value 2.5")
                
                # Calculate TRI (4-20 scale)
                cohort_df['TRI'] = (cohort_df['Pedagogical_Confidence'] + 
                                    cohort_df['Digital_Literacy'] + 
                                    cohort_df['Resource_Availability'] + 
                                    cohort_df['Training_Quality'])
                
                # Generate summary statistics
                st.markdown("###  Cohort Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean TRI", f"{cohort_df['TRI'].mean():.2f}")
                with col2:
                    st.metric("Median TRI", f"{cohort_df['TRI'].median():.2f}")
                with col3:
                    st.metric("Min TRI", f"{cohort_df['TRI'].min():.2f}")
                with col4:
                    st.metric("Max TRI", f"{cohort_df['TRI'].max():.2f}")
                
                # Distribution plot
                st.markdown("###  TRI Distribution")
                fig = px.histogram(
                    cohort_df, 
                    x='TRI', 
                    nbins=20,
                    title='Distribution of Readiness Scores in Cohort',
                    labels={'TRI': 'Teacher Readiness Index (4-20 scale)'},
                    color_discrete_sequence=['#2a5298']
                )
                fig.add_vline(x=cohort_df['TRI'].mean(), line_dash="dash", line_color="red", 
                             annotation_text=f"Mean: {cohort_df['TRI'].mean():.2f}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Readiness bands
                st.markdown("### 🎯 Readiness Band Distribution")
                
                conditions = [
                    (cohort_df['TRI'] < 10),
                    (cohort_df['TRI'] >= 10) & (cohort_df['TRI'] < 15),
                    (cohort_df['TRI'] >= 15)
                ]
                choices = ['Low Readiness', 'Moderate Readiness', 'High Readiness']
                cohort_df['Readiness_Band'] = np.select(conditions, choices, default='Unknown')
                
                band_counts = cohort_df['Readiness_Band'].value_counts().reset_index()
                band_counts.columns = ['Band', 'Count']
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig = px.pie(
                        band_counts, 
                        values='Count', 
                        names='Band',
                        title='Readiness Band Distribution',
                        color='Band',
                        color_discrete_map={
                            'Low Readiness': '#dc3545',
                            'Moderate Readiness': '#ffc107',
                            'High Readiness': '#28a745'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        band_counts, 
                        x='Band', 
                        y='Count',
                        title='Teachers by Readiness Band',
                        color='Band',
                        color_discrete_map={
                            'Low Readiness': '#dc3545',
                            'Moderate Readiness': '#ffc107',
                            'High Readiness': '#28a745'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Construct-level analysis
                st.markdown("### 🔧 Construct-Level Analysis")
                
                # Prepare data for construct comparison
                construct_data = pd.DataFrame({
                    'Construct': ['Pedagogical', 'Digital', 'Resources', 'Training'],
                    'Mean Score': [
                        cohort_df['Pedagogical_Confidence'].mean(),
                        cohort_df['Digital_Literacy'].mean(),
                        cohort_df['Resource_Availability'].mean(),
                        cohort_df['Training_Quality'].mean()
                    ]
                })
                
                fig = px.bar(
                    construct_data,
                    x='Construct',
                    y='Mean Score',
                    title='Average Construct Scores Across Cohort',
                    color='Mean Score',
                    color_continuous_scale='Viridis',
                    range_y=[1, 5]
                )
                fig.add_hline(y=3, line_dash="dash", line_color="red", 
                             annotation_text="Moderate Threshold (3.0)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                st.markdown("###  Export Results")
                
                csv_full = cohort_df.to_csv(index=False)
                b64_full = base64.b64encode(csv_full.encode()).decode()
                href_full = f'<a href="data:file/csv;base64,{b64_full}" download="cohort_analysis.csv"> Download Full Analysis (CSV)</a>'
                st.markdown(href_full, unsafe_allow_html=True)
            
            else:
                st.error("""
                ❌ Could not find any matching columns in your CSV.
                
                Required columns should include: conf_B2, conf_B6, digi_C1c, digi_C1d, digi_C2e, 
                res_D1 through res_D6, and train_E2 through train_E7.
                
                Please download the template and ensure your column names match exactly.
                """)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
# ============================================
# TAB 4: PROGRESS TRACKING (Keep as is)
# ============================================
with tab4:
    st.markdown("##  Progress Tracking")
    
    if st.session_state.previous_predictions:
        history_df = pd.DataFrame(st.session_state.previous_predictions)
        
        # Show history
        st.markdown("### Assessment History")
        st.dataframe(history_df)
        
        # Trend chart
        fig = px.line(history_df, x=history_df.index, y='score',
                      title='TRI Score Trend',
                      labels={'index': 'Assessment Number', 'score': 'TRI Score'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        st.markdown("### Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Assessments", len(history_df))
        with col2:
            st.metric("Average", f"{history_df['score'].mean():.1f}")
        with col3:
            st.metric("Improvement", 
                     f"{history_df['score'].iloc[-1] - history_df['score'].iloc[0]:.1f}" 
                     if len(history_df) > 1 else "N/A")
    else:
        st.info("No assessments recorded yet. Complete an assessment in the Individual Assessment tab.")

# ============================================
# TAB 5: TECHNICAL DOCUMENTATION (Keep as is)
# ============================================
with tab5:
    st.markdown("##  Technical Documentation")
    
    with st.expander(" Model Performance", expanded=True):
        st.markdown("""
        **Final Stacking Ensemble Performance:**
        - R² Score: **0.9763** on held-out test set
        - Mean Absolute Error: **0.245** points
        - 87.6% of predictions within ±0.5 points
        - 100% of predictions within ±1.0 points
        
        **Model Architecture:**
        - Base Learners: MLR, Ridge Regression, XGBoost, SVR
        - Meta-Learner: Ridge Regression
        - 5-fold GroupKFold cross-validation (school-grouped)
        """)
        
        # Model weights visualization
        weights_df = pd.DataFrame({
            'Model': ['SVR', 'XGBoost', 'Ridge', 'MLR'],
            'Weight': [38.5, 27.7, 18.3, 16.5]
        })
        
        fig = px.pie(weights_df, values='Weight', names='Model',
                     title='Ensemble Model Weights')
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander(" Feature Importance"):
        st.markdown("""
        **Top Predictors:**
        1. conf_B2 (Pedagogical Confidence - Classroom Management): β = 0.461
        2. conf_B6 (Pedagogical Confidence - Learner Engagement): β = 0.452
        3. digi_C1c (Digital Literacy - Content Creation): β = 0.308
        4. digi_C1d (Digital Literacy - Digital Assessment): β = 0.298
        5. digi_C2e (Digital Literacy - Collaboration): β = 0.257
        """)
    
    with st.expander(" Key Research Findings"):
        st.markdown("""
        **Main Findings:**
        1. **Pedagogical Confidence** is the strongest predictor of readiness
        2. **Training Quality** matters more than attendance (p = 0.364 for attendance)
        3. **Digital Literacy** only contributes meaningfully when Training Quality > 3.0
        4. ~11% of teachers show **confidence-skills mismatch**
        5. Linear relationships dominate (MLR R² = 0.9699)
        """)
    
    with st.expander(" References"):
        st.markdown("""
        - Mishra, P., & Koehler, M. J. (2006). Technological Pedagogical Content Knowledge: A framework for teacher knowledge.
        - Bandura, A. (1997). Self-efficacy: The exercise of control.
        - Fullan, M. (2015). The new meaning of educational change.
        - Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2026 Strathmore University - Institute of Mathematical Sciences</p>
    <p>Developed as part of MSc in Data Science and Analytics Thesis</p>
    <p>Principal Researcher: Peter Wachugu Kanyuira | Supervisor: Dr. John Olukuru</p>
</div>
""", unsafe_allow_html=True)