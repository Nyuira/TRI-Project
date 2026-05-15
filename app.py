"""
============================================================
CBC TEACHER READINESS DIAGNOSTIC DASHBOARD
Version 3.0 — Full Policy Intelligence Edition

MSc Data Science and Analytics — Strathmore University
Principal Researcher: Peter Wachugu Kanyuira

Tabs:
  1. Individual Assessment
  2. Intervention Simulator
  3. School Dashboard (Principal)
  4. County Dashboard (County Education Officer)
  5. Policy Intelligence (Ministry of Education)
  6. Cohort Analysis
  7. Progress Tracking
  8. Technical Documentation
============================================================
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CBC Teacher Readiness Diagnostic",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
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
.danger-box {
    background: #f8d7da;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #dc3545;
    margin: 1rem 0;
}
.policy-box {
    background: #e8f4f8;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #17a2b8;
    margin: 1rem 0;
}
.success-box {
    background: #d4edda;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin: 1rem 0;
}
.section-header {
    color: #1e3c72;
    font-weight: 600;
    margin: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 0.5rem;
}
.score-badge {
    font-size: 3rem;
    font-weight: 700;
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
.mismatch-alert {
    background: #fff3cd;
    border: 2px solid #ffc107;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.roadmap-step {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #2a5298;
}
.footer {
    text-align: center;
    padding: 2rem;
    color: #6c757d;
    font-size: 0.9rem;
}
.sdg-badge {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 15px;
    font-size: 0.85rem;
    font-weight: 600;
}
.traffic-green {
    background: #28a745;
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 25px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}
.traffic-amber {
    background: #ffc107;
    color: #212529;
    padding: 0.5rem 1.5rem;
    border-radius: 25px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}
.traffic-red {
    background: #dc3545;
    color: white;
    padding: 0.5rem 1.5rem;
    border-radius: 25px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALISATION
# ============================================================
defaults = {
    'predictions_made': 0,
    'total_score': 0,
    'previous_predictions': [],
    'last_result': None,
    'last_constructs': None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_readiness_band(score):
    if score < 10:
        return "Low Readiness", "band-low", "#dc3545"
    elif score < 15:
        return "Moderate Readiness", "band-moderate", "#ffc107"
    else:
        return "High Readiness", "band-high", "#28a745"


def get_percentile(score):
    """Approximate percentile based on sample distribution
    (mean=11.46, SD=1.85, Nakuru County norms)."""
    z = (score - 11.46) / 1.85
    if z < -2.0:   return 2
    elif z < -1.5: return 7
    elif z < -1.0: return 16
    elif z < -0.5: return 31
    elif z < 0.0:  return 50
    elif z < 0.5:  return 69
    elif z < 1.0:  return 84
    elif z < 1.5:  return 93
    else:          return 98


def detect_mismatch(ped_mean, dig_mean, res_mean, train_mean):
    """
    Detect confidence-skills mismatch profile.
    Returns: ('overconfident' | 'underconfident' | 'aligned', description)
    """
    skill_mean = (dig_mean + res_mean + train_mean) / 3
    gap = ped_mean - skill_mean
    if gap > 1.2:
        return 'overconfident', (
            "⚠️ **Overconfident Profile Detected**\n\n"
            "This teacher reports high Pedagogical Confidence "
            f"({ped_mean:.1f}/5) but lower Digital Literacy, Resource "
            f"Availability, and Training Quality (average {skill_mean:.1f}/5). "
            "Research shows this group may not seek support despite skill gaps. "
            "**Recommended action:** Schedule a structured skill audit and "
            "provide targeted, evidence-based skill-building workshops."
        )
    elif gap < -1.2:
        return 'underconfident', (
            "💡 **Underconfident Profile Detected**\n\n"
            "This teacher has solid Digital Literacy, Resource access, and "
            f"Training Quality (average {skill_mean:.1f}/5) but reports low "
            f"Pedagogical Confidence ({ped_mean:.1f}/5). This represents "
            "unrealised capacity. "
            "**Recommended action:** Provide mastery experiences — structured "
            "opportunities to lead lessons and receive positive peer feedback — "
            "to build self-efficacy."
        )
    else:
        return 'aligned', None


def generate_tiered_roadmap(ped, dig, res, train, band):
    """Generate a 3/6/12-month roadmap based on score profile."""
    roadmap = {}

    # Identify priority areas
    scores = {
        'Pedagogical Confidence': ped,
        'Digital Literacy': dig,
        'Resource Availability': res,
        'Training Quality': train
    }
    sorted_areas = sorted(scores.items(), key=lambda x: x[1])
    priority_1 = sorted_areas[0][0]
    priority_2 = sorted_areas[1][0]

    # Threshold check
    threshold_warning = ""
    if train < 3.0 and dig > 3.0:
        threshold_warning = (
            "⚠️ **Training Quality Threshold Alert:** Digital upskilling will "
            "have limited impact until Training Quality exceeds 3.0. "
            "Prioritise training quality improvement first."
        )

    actions = {
        'Pedagogical Confidence': {
            '3m': "Participate in at least two peer observation sessions. "
                  "Request structured feedback from head of department.",
            '6m': "Lead one demonstration lesson in your school. "
                  "Enrol in a CBC pedagogy coaching programme.",
            '12m': "Mentor one junior colleague. Complete a reflective "
                   "teaching portfolio documenting CBC lesson delivery."
        },
        'Digital Literacy': {
            '3m': "Complete a hands-on digital tools workshop "
                  "(e.g., Google Classroom, Kahoot, Canva for Education).",
            '6m': "Integrate at least one digital tool into weekly lesson "
                  "planning. Share a digital resource with your department.",
            '12m': "Design and deliver a fully digitally integrated CBC unit. "
                   "Assess student competencies using a digital tool."
        },
        'Resource Availability': {
            '3m': "Document specific resource gaps (ICT, materials, "
                  "connectivity) and present a needs report to your principal.",
            '6m': "Explore community partnerships or NGO grants for "
                  "resource provision. Identify low-cost digital alternatives.",
            '12m': "Advocate through school governing board for a "
                   "dedicated CBC resource budget line in the school plan."
        },
        'Training Quality': {
            '3m': "Identify and register for a high-quality, school-contextualised "
                  "CBC professional development programme (not Ministry attendance only).",
            '6m': "Complete subject-specific CBC training with follow-up "
                  "coaching. Document training quality using the TRI survey.",
            '12m': "Establish or join a peer professional learning community "
                   "with monthly structured CBC practice sharing sessions."
        }
    }

    roadmap['threshold_warning'] = threshold_warning
    roadmap['priority_1'] = priority_1
    roadmap['priority_2'] = priority_2
    roadmap['3_month'] = [
        actions[priority_1]['3m'],
        actions[priority_2]['3m']
    ]
    roadmap['6_month'] = [
        actions[priority_1]['6m'],
        actions[priority_2]['6m']
    ]
    roadmap['12_month'] = [
        actions[priority_1]['12m'],
        actions[priority_2]['12m']
    ]

    if band == "High Readiness":
        roadmap['3_month'].append(
            "Serve as a peer mentor for at least one colleague "
            "in the Low or Moderate readiness band."
        )
        roadmap['6_month'].append(
            "Lead a school-wide CBC best-practice sharing session."
        )
        roadmap['12_month'].append(
            "Document your CBC implementation journey as a case study "
            "to be shared at county level."
        )

    return roadmap


def build_shap_waterfall(ped, dig, res, train, prediction,
                         base_value=11.46):
    """
    Approximate SHAP waterfall using MLR coefficients from the study.
    Coefficient-based marginal contributions relative to sample means.
    Sample means: ped=3.19, dig=2.66, res=2.85, train=2.76
    """
    sample_means = {'ped': 3.19, 'dig': 2.66, 'res': 2.85, 'train': 2.76}
    # Approximate weights from SHAP global importance
    weights = {'ped': 0.461 * 2, 'dig': 0.308 * 3,
               'res': 0.204 * 6, 'train': 0.207 * 6}

    contribs = {
        'Pedagogical\nConfidence': weights['ped'] * (ped - sample_means['ped']),
        'Digital\nLiteracy':       weights['dig'] * (dig - sample_means['dig']),
        'Resource\nAvailability':  weights['res'] * (res - sample_means['res']),
        'Training\nQuality':       weights['train'] * (train - sample_means['train']),
    }

    # Scale so sum matches prediction - base_value
    total = sum(contribs.values())
    target = prediction - base_value
    if abs(total) > 0.001:
        scale = target / total
        contribs = {k: v * scale for k, v in contribs.items()}

    features = list(contribs.keys())
    values   = list(contribs.values())
    colors   = ['#e74c3c' if v >= 0 else '#3498db' for v in values]

    # Build waterfall: running base
    running = base_value
    bases   = []
    for v in values:
        bases.append(running)
        running += v

    fig = go.Figure()
    for i, (feat, val, base, col) in enumerate(
            zip(features, values, bases, colors)):
        fig.add_trace(go.Bar(
            x=[feat],
            y=[abs(val)],
            base=[base if val >= 0 else base + val],
            marker_color=col,
            name='+' if val >= 0 else '−',
            showlegend=False,
            text=[f"{val:+.2f}"],
            textposition='outside'
        ))

    # Base and final lines
    fig.add_hline(y=base_value, line_dash='dot', line_color='grey',
                  annotation_text=f'Base (mean): {base_value}',
                  annotation_position='right')
    fig.add_hline(y=prediction, line_dash='dash', line_color='#1e3c72',
                  annotation_text=f'Predicted TRI: {prediction:.1f}',
                  annotation_position='right')

    fig.update_layout(
        title='SHAP Waterfall — Construct Contributions to TRI Score',
        yaxis_title='TRI Score',
        xaxis_title='Construct',
        height=420,
        showlegend=False,
        plot_bgcolor='white'
    )
    return fig


def tri_from_constructs(ped, dig, res, train):
    return (ped + dig + res + train) * 4 / 4


def resource_gap_analysis(res_D1, res_D2, res_D3, res_D4, res_D5, res_D6):
    items = {
        'ICT Infrastructure': res_D1,
        'Internet Connectivity': res_D2,
        'Learning Materials': res_D3,
        'Technical Support': res_D4,
        'Electricity Reliability': res_D5,
        'Equipment Maintenance': res_D6,
    }
    return dict(sorted(items.items(), key=lambda x: x[1]))


def generate_csv_download(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return (f'<a href="data:file/csv;base64,{b64}" '
            f'download="{filename}">📥 Download {filename}</a>')


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('07_final_stacking_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file '07_final_stacking_model.pkl' not found. "
                 "Please place the model file in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


final_model = load_model()

FEATURE_ORDER = [
    'train_E4', 'train_E3', 'train_E7', 'train_E6', 'train_E2', 'train_E5',
    'Digital_x_Training', 'res_D6', 'training_attended_binary', 'res_D5',
    'res_D4', 'digi_C1d', 'res_D3', 'res_D1', 'res_D2', 'digi_C1c',
    'digi_C2e', 'conf_B2', 'conf_B6'
]

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">🎓 CBC Teacher Readiness Diagnostic</h1>
    <p style="margin:0; opacity:0.9">
        Evidence-based decision support for Competency-Based Curriculum implementation
    </p>
    <p style="margin:0; font-size:0.85rem; margin-top:0.5rem">
        Version 3.0 — Policy Intelligence Edition &nbsp;|&nbsp;
        Strathmore University — MSc Data Science and Analytics &nbsp;|&nbsp;
        Nakuru County Norms: Mean = 11.46, SD = 1.85
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 👤 User Role")
    role = st.selectbox(
        "I am a:",
        ["Individual Teacher", "School Principal",
         "County Education Officer", "Ministry Official", "Researcher"],
        help="Select your role to see role-relevant guidance throughout the dashboard."
    )

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Diagnostics Run", st.session_state.predictions_made)
    with c2:
        if st.session_state.predictions_made > 0:
            avg = st.session_state.total_score / st.session_state.predictions_made
            st.metric("Avg TRI", f"{avg:.1f}")

    st.markdown("---")
    st.markdown("### 📥 Export Session")
    if st.button("Export All Session Data", use_container_width=True):
        if st.session_state.previous_predictions:
            export_df = pd.DataFrame(st.session_state.previous_predictions)
            st.markdown(
                generate_csv_download(export_df, "session_data.csv"),
                unsafe_allow_html=True
            )
        else:
            st.info("No data yet.")

    st.markdown("---")
    st.markdown("### 🔑 Key Finding")
    st.info(
        "Digital Literacy only improves TRI when Training Quality > 3.0. "
        "Prioritise training quality before digital upskilling."
    )
    st.markdown("---")
    st.caption("© 2026 Strathmore University\n"
               "Researcher: Peter Wachugu Kanyuira")

# ============================================================
# TABS
# ============================================================
(tab1, tab2, tab3, tab4, tab5,
 tab6, tab7, tab8) = st.tabs([
    "👩‍🏫 Individual Assessment",
    "🔬 Intervention Simulator",
    "🏫 School Dashboard",
    "🏛️ County Dashboard",
    "📋 Policy Intelligence",
    "📊 Cohort Analysis",
    "📈 Progress Tracking",
    "⚙️ Technical Docs"
])

# ============================================================
# TAB 1 — INDIVIDUAL ASSESSMENT
# ============================================================
with tab1:
    st.markdown("## 👩‍🏫 Individual Teacher Readiness Assessment")
    st.markdown(
        "Complete all four sections below to generate a personalised "
        "readiness profile, SHAP diagnostic chart, mismatch detection, "
        "and a three-phase development roadmap."
    )

    sections_done = 0
    left, right = st.columns([2, 1])

    with left:
        # --- Section 1: Pedagogical Confidence ---
        with st.expander("🧠 Section 1: Pedagogical Confidence", expanded=True):
            st.caption("How confident are you in these areas of CBC delivery?")
            c1, c2 = st.columns(2)
            label_15 = lambda x: ["Very Low","Low","Moderate","High","Very High"][x-1]
            with c1:
                conf_B2 = st.selectbox(
                    "Classroom Management", [1,2,3,4,5], index=2,
                    format_func=label_15, key="c_B2",
                    help="Ability to manage a learner-centred CBC classroom"
                )
            with c2:
                conf_B6 = st.selectbox(
                    "Learner Engagement", [1,2,3,4,5], index=2,
                    format_func=label_15, key="c_B6",
                    help="Confidence in engaging students in active learning"
                )
            sections_done += 1

        # --- Section 2: Digital Literacy ---
        with st.expander("💻 Section 2: Digital Literacy", expanded=True):
            st.caption("Rate your digital teaching capabilities.")
            c1, c2, c3 = st.columns(3)
            with c1:
                digi_C1c = st.selectbox(
                    "Content Creation", [1,2,3,4,5], index=2,
                    format_func=label_15, key="d_C1c",
                    help="Creating digital content for CBC lessons"
                )
            with c2:
                digi_C1d = st.selectbox(
                    "Digital Assessment", [1,2,3,4,5], index=2,
                    format_func=label_15, key="d_C1d",
                    help="Using digital tools for CBC-aligned assessment"
                )
            with c3:
                digi_C2e = st.selectbox(
                    "Digital Collaboration", [1,2,3,4,5], index=2,
                    format_func=label_15, key="d_C2e",
                    help="Facilitating online or technology-mediated collaboration"
                )
            sections_done += 1

        # --- Section 3: Resource Availability ---
        with st.expander("🏗️ Section 3: Resource Availability", expanded=True):
            st.caption("Assess your school's CBC resources.")
            r_labels = {
                'res_D1': ("ICT Infrastructure",
                           ["Very Inadequate","Inadequate","Adequate","Good","Excellent"]),
                'res_D2': ("Internet Connectivity",
                           ["Very Poor","Poor","Fair","Good","Excellent"]),
                'res_D3': ("Learning Materials",
                           ["Very Inadequate","Inadequate","Adequate","Good","Excellent"]),
                'res_D4': ("Technical Support",
                           ["Very Poor","Poor","Fair","Good","Excellent"]),
                'res_D5': ("Electricity Reliability",
                           ["Very Unreliable","Unreliable","Fair","Reliable","Very Reliable"]),
                'res_D6': ("Equipment Maintenance",
                           ["Very Poor","Poor","Fair","Good","Excellent"]),
            }
            c1, c2 = st.columns(2)
            r_vals = {}
            keys = list(r_labels.keys())
            for i, k in enumerate(keys):
                lbl, opts = r_labels[k]
                col = c1 if i < 3 else c2
                with col:
                    r_vals[k] = st.radio(
                        lbl, [1,2,3,4,5], index=2,
                        horizontal=True,
                        format_func=lambda x, o=opts: o[x-1],
                        key=f"r_{k}"
                    )
            sections_done += 1
            res_D1,res_D2,res_D3 = r_vals['res_D1'],r_vals['res_D2'],r_vals['res_D3']
            res_D4,res_D5,res_D6 = r_vals['res_D4'],r_vals['res_D5'],r_vals['res_D6']

        # --- Section 4: Training Quality ---
        with st.expander("📚 Section 4: Training Quality", expanded=True):
            st.caption("Evaluate the quality of your CBC professional development.")
            train_attended = st.checkbox(
                "I have attended CBC professional development training",
                value=True, key="t_att"
            )
            t_label = lambda x: ["Very Poor","Poor","Fair","Good","Excellent"][x-1]
            if train_attended:
                c1, c2 = st.columns(2)
                with c1:
                    train_E2 = st.selectbox("Training Comprehensiveness",
                        [1,2,3,4,5],index=2,format_func=t_label,key="t_E2")
                    train_E3 = st.selectbox("Practical Applicability",
                        [1,2,3,4,5],index=2,format_func=t_label,key="t_E3")
                    train_E4 = st.selectbox("Trainer Effectiveness",
                        [1,2,3,4,5],index=2,format_func=t_label,key="t_E4")
                with c2:
                    train_E5 = st.selectbox("Practice Opportunities",
                        [1,2,3,4,5],index=2,format_func=t_label,key="t_E5")
                    train_E6 = st.selectbox("Follow-up Support",
                        [1,2,3,4,5],index=2,format_func=t_label,key="t_E6")
                    train_E7 = st.selectbox("Confidence Improvement",
                        [1,2,3,4,5],index=2,format_func=t_label,key="t_E7")
                training_attended_binary = 1.0
            else:
                train_E2=train_E3=train_E4=train_E5=train_E6=train_E7 = 1
                training_attended_binary = 0.0
                st.info("Training quality defaults to 1 (Very Poor) — "
                        "consistent with the study's logical imputation.")
            sections_done += 1

        _, bcol, _ = st.columns([1,1,1])
        with bcol:
            run_btn = st.button(
                "🔍 Generate Readiness Profile",
                width='stretch', type="primary", key="gen_t1"
            )

    with right:
        st.markdown("### ✅ Progress")
        st.progress(sections_done / 4)
        st.markdown(f"**{sections_done}/4 sections complete**")
        if sections_done == 4:
            st.success("Ready for analysis.")
        st.markdown("---")
        st.markdown("### 📌 Tips")
        st.info("Answer based on your current situation, not aspirational targets. "
                "Honest responses produce the most useful diagnostic output.")

    # ---- RESULTS ----
    if run_btn:
        train_mean_val = (train_E2+train_E3+train_E4+train_E5+train_E6+train_E7)/6
        dig_x_train    = float(digi_C1c) * train_mean_val

        data = {
            'train_E4': float(train_E4), 'train_E3': float(train_E3),
            'train_E7': float(train_E7), 'train_E6': float(train_E6),
            'train_E2': float(train_E2), 'train_E5': float(train_E5),
            'Digital_x_Training': dig_x_train,
            'res_D6': float(res_D6), 'training_attended_binary': training_attended_binary,
            'res_D5': float(res_D5), 'res_D4': float(res_D4),
            'digi_C1d': float(digi_C1d), 'res_D3': float(res_D3),
            'res_D1': float(res_D1),  'res_D2': float(res_D2),
            'digi_C1c': float(digi_C1c), 'digi_C2e': float(digi_C2e),
            'conf_B2': float(conf_B2), 'conf_B6': float(conf_B6)
        }

        input_df = pd.DataFrame([data])[FEATURE_ORDER]
        prediction = float(np.clip(final_model.predict(input_df)[0], 4.0, 20.0))

        # Construct means
        ped_m   = (conf_B2 + conf_B6) / 2
        dig_m   = (digi_C1c + digi_C1d + digi_C2e) / 3
        res_m   = (res_D1+res_D2+res_D3+res_D4+res_D5+res_D6) / 6
        train_m = (train_E2+train_E3+train_E4+train_E5+train_E6+train_E7) / 6

        # Update session
        st.session_state.predictions_made += 1
        st.session_state.total_score      += prediction
        st.session_state.last_result       = prediction
        st.session_state.last_constructs   = {
            'ped': ped_m, 'dig': dig_m, 'res': res_m, 'train': train_m
        }
        st.session_state.previous_predictions.append({
            'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
            'TRI_Score': round(prediction, 2),
            'Ped_Confidence': round(ped_m, 2),
            'Digital_Literacy': round(dig_m, 2),
            'Resource_Availability': round(res_m, 2),
            'Training_Quality': round(train_m, 2),
        })

        band, band_class, band_colour = get_readiness_band(prediction)
        pct = get_percentile(prediction)

        st.markdown("---")
        st.markdown("## 📊 Readiness Assessment Results")

        # Key metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("TRI Score (4–20)", f"{prediction:.1f}")
        with m2:
            st.metric("Readiness Band", band)
        with m3:
            st.metric("Approx. Percentile", f"~{pct}th")
        with m4:
            dist_from_threshold = 11.46 - prediction
            st.metric("vs County Mean",
                      f"{'+' if prediction >= 11.46 else ''}"
                      f"{prediction - 11.46:.1f} pts")

        # --- SHAP Waterfall ---
        st.markdown("### 📉 SHAP Diagnostic Waterfall")
        st.caption(
            "Each bar shows how much a construct pushed your TRI score "
            "above (red) or below (blue) the county average of 11.46."
        )
        fig_shap = build_shap_waterfall(ped_m, dig_m, res_m, train_m, prediction)
        st.plotly_chart(fig_shap, use_container_width=True)

        # --- Radar chart ---
        st.markdown("### 🕸️ Construct Profile Radar")
        fig_radar = go.Figure()
        cats = ['Pedagogical\nConfidence','Digital\nLiteracy',
                'Resource\nAvailability','Training\nQuality']
        fig_radar.add_trace(go.Scatterpolar(
            r=[ped_m, dig_m, res_m, train_m],
            theta=cats, fill='toself', name='Your Profile',
            line_color='#2a5298', fillcolor='rgba(42,82,152,0.3)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[3.0, 3.0, 3.0, 3.0], theta=cats,
            fill='none', name='Moderate Threshold',
            line=dict(color='red', dash='dash')
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1,5])),
            showlegend=True, height=380,
            title="Readiness Profile vs Moderate Threshold (3.0)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # --- Mismatch Detection ---
        mismatch_type, mismatch_msg = detect_mismatch(ped_m, dig_m, res_m, train_m)
        if mismatch_type != 'aligned':
            st.markdown("### 🔍 Mismatch Profile Alert")
            st.markdown(
                f'<div class="mismatch-alert">{mismatch_msg}</div>',
                unsafe_allow_html=True
            )

        # --- Training Threshold Warning ---
        if train_m < 3.0:
            st.markdown(
                '<div class="warning-box">'
                '<strong>⚠️ Training Quality Threshold Alert</strong><br>'
                f'Your Training Quality score is {train_m:.1f}/5 — below the '
                'critical threshold of 3.0. Research findings show that Digital '
                'Literacy improvements yield minimal TRI gains until Training Quality '
                'exceeds 3.0. <strong>Prioritise foundational CBC training quality '
                'before digital upskilling.</strong>'
                '</div>',
                unsafe_allow_html=True
            )

        # --- Resource Gap Highlight ---
        res_gaps = resource_gap_analysis(
            res_D1, res_D2, res_D3, res_D4, res_D5, res_D6)
        worst_resource = list(res_gaps.keys())[0]
        worst_score    = list(res_gaps.values())[0]
        if worst_score < 3:
            st.markdown(
                f'<div class="warning-box">'
                f'<strong>🏗️ Critical Resource Gap Identified</strong><br>'
                f'<em>{worst_resource}</em> is your lowest-rated resource '
                f'({worst_score}/5). Advocate with your principal or county '
                f'officer to address this gap as a priority action.'
                f'</div>',
                unsafe_allow_html=True
            )

        # --- Tiered Roadmap ---
        st.markdown("### 🗺️ Personalised Development Roadmap")
        roadmap = generate_tiered_roadmap(ped_m, dig_m, res_m, train_m, band)

        if roadmap['threshold_warning']:
            st.warning(roadmap['threshold_warning'])

        col_3m, col_6m, col_12m = st.columns(3)
        with col_3m:
            st.markdown("#### 📅 0–3 Months")
            st.markdown("*Priority interventions:*")
            for action in roadmap['3_month']:
                st.markdown(
                    f'<div class="roadmap-step">📌 {action}</div>',
                    unsafe_allow_html=True
                )
        with col_6m:
            st.markdown("#### 📅 3–6 Months")
            st.markdown("*Skill consolidation:*")
            for action in roadmap['6_month']:
                st.markdown(
                    f'<div class="roadmap-step">🔧 {action}</div>',
                    unsafe_allow_html=True
                )
        with col_12m:
            st.markdown("#### 📅 6–12 Months")
            st.markdown("*Leadership and embedding:*")
            for action in roadmap['12_month']:
                st.markdown(
                    f'<div class="roadmap-step">🏆 {action}</div>',
                    unsafe_allow_html=True
                )

        # --- Export Personal Development Plan ---
        st.markdown("### 📥 Download Personal Development Plan")
        plan_data = {
            'Field': [
                'TRI Score', 'Readiness Band', 'Approx. Percentile',
                'Pedagogical Confidence', 'Digital Literacy',
                'Resource Availability', 'Training Quality',
                'Mismatch Type',
                '0-3 Month Action 1', '0-3 Month Action 2',
                '3-6 Month Action 1', '3-6 Month Action 2',
                '6-12 Month Action 1', '6-12 Month Action 2'
            ],
            'Value': [
                f"{prediction:.1f}", band, f"~{pct}th",
                f"{ped_m:.2f}/5", f"{dig_m:.2f}/5",
                f"{res_m:.2f}/5", f"{train_m:.2f}/5",
                mismatch_type.title(),
                roadmap['3_month'][0],
                roadmap['3_month'][1] if len(roadmap['3_month'])>1 else '',
                roadmap['6_month'][0],
                roadmap['6_month'][1] if len(roadmap['6_month'])>1 else '',
                roadmap['12_month'][0],
                roadmap['12_month'][1] if len(roadmap['12_month'])>1 else '',
            ]
        }
        plan_df = pd.DataFrame(plan_data)
        st.markdown(
            generate_csv_download(plan_df, "personal_development_plan.csv"),
            unsafe_allow_html=True
        )

# ============================================================
# TAB 2 — INTERVENTION SIMULATOR
# ============================================================
with tab2:
    st.markdown("## 🔬 Intervention Impact Simulator")
    st.markdown(
        "Adjust target scores to model the expected TRI impact "
        "of specific professional development interventions. "
        "The simulator applies the model's construct weightings."
    )

    st.markdown(
        '<div class="policy-box">'
        '<strong>📋 Research-Based Sequencing Guidance</strong><br>'
        'Evidence from this study shows that Digital Literacy improvements '
        'yield minimal TRI gains when Training Quality is below 3.0. '
        'The simulator will flag this condition automatically.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Current Profile")
        cur_ped   = st.slider("Current Pedagogical Confidence", 1.0,5.0,2.5,0.5,key="sim_cp")
        cur_dig   = st.slider("Current Digital Literacy",       1.0,5.0,2.5,0.5,key="sim_cd")
        cur_res   = st.slider("Current Resource Availability",  1.0,5.0,2.5,0.5,key="sim_cr")
        cur_train = st.slider("Current Training Quality",       1.0,5.0,2.5,0.5,key="sim_ct")

    with c2:
        st.markdown("### Target Profile After Intervention")
        tgt_ped   = st.slider("Target Pedagogical Confidence", 1.0,5.0,3.5,0.5,key="sim_tp")
        tgt_dig   = st.slider("Target Digital Literacy",       1.0,5.0,3.5,0.5,key="sim_td")
        tgt_res   = st.slider("Target Resource Availability",  1.0,5.0,3.5,0.5,key="sim_tr")
        tgt_train = st.slider("Target Training Quality",       1.0,5.0,3.5,0.5,key="sim_tt")

    # Cost-Benefit Estimator
    st.markdown("### 💰 Resource Allocation Estimator")
    st.caption(
        "Enter an approximate budget to estimate the highest-return intervention."
    )
    budget = st.number_input(
        "Available Professional Development Budget (KSh)", 
        min_value=0, value=50000, step=5000, key="budget_input"
    )

    # Approximate cost per improvement unit by construct
    cost_per_unit = {
        'Pedagogical Confidence': 8000,   # coaching sessions
        'Digital Literacy': 12000,         # device + training
        'Resource Availability': 25000,    # procurement
        'Training Quality': 6000           # quality PD programme
    }

    if st.button("⚡ Simulate Intervention Impact", type="primary", key="sim_btn"):
        cur_tri = tri_from_constructs(cur_ped, cur_dig, cur_res, cur_train)
        tgt_tri = tri_from_constructs(tgt_ped, tgt_dig, tgt_res, tgt_train)
        gain    = tgt_tri - cur_tri

        # Per-construct gains
        gains = {
            'Pedagogical Confidence': (tgt_ped   - cur_ped)   * 0.461 * 2,
            'Digital Literacy':       (tgt_dig   - cur_dig)   * 0.308 * 3,
            'Resource Availability':  (tgt_res   - cur_res)   * 0.204 * 6,
            'Training Quality':       (tgt_train - cur_train) * 0.207 * 6,
        }

        # Key metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Current TRI",  f"{cur_tri:.1f}")
        with m2:
            st.metric("Target TRI",   f"{tgt_tri:.1f}")
        with m3:
            delta_col = "normal" if gain >= 0 else "inverse"
            st.metric("Expected Gain", f"{gain:+.1f}", delta_color=delta_col)

        # Gain chart
        gain_df = pd.DataFrame({
            'Construct': list(gains.keys()),
            'Expected TRI Gain': list(gains.values()),
            'Current Score': [cur_ped, cur_dig, cur_res, cur_train],
            'Target Score':  [tgt_ped, tgt_dig, tgt_res, tgt_train],
        })

        fig = px.bar(
            gain_df, x='Construct', y='Expected TRI Gain',
            color='Expected TRI Gain',
            color_continuous_scale='RdYlGn',
            title='Expected TRI Gain by Construct Intervention',
            text=gain_df['Expected TRI Gain'].round(2)
        )
        fig.add_hline(y=0, line_color='grey', line_dash='dot')
        st.plotly_chart(fig, use_container_width=True)

        # Before/After grouped bar
        fig2 = go.Figure(data=[
            go.Bar(name='Current',
                   x=['Pedagogical','Digital','Resources','Training'],
                   y=[cur_ped, cur_dig, cur_res, cur_train],
                   marker_color='#6c757d'),
            go.Bar(name='Target',
                   x=['Pedagogical','Digital','Resources','Training'],
                   y=[tgt_ped, tgt_dig, tgt_res, tgt_train],
                   marker_color='#2a5298')
        ])
        fig2.update_layout(
            barmode='group', title='Current vs Target Construct Scores', height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Threshold check
        if cur_train < 3.0 and tgt_dig > cur_dig:
            st.markdown(
                '<div class="warning-box">'
                '⚠️ <strong>Threshold Alert:</strong> Digital Literacy improvement '
                'will have limited TRI impact because Training Quality '
                f'({cur_train:.1f}) is below 3.0. '
                'Consider allocating budget to Training Quality first.'
                '</div>',
                unsafe_allow_html=True
            )

        # Cost-Benefit Recommendation
        st.markdown("### 💡 Budget Allocation Recommendation")
        improvements = {
            'Pedagogical Confidence': tgt_ped   - cur_ped,
            'Digital Literacy':       tgt_dig   - cur_dig,
            'Resource Availability':  tgt_res   - cur_res,
            'Training Quality':       tgt_train - cur_train,
        }
        roi = {}
        for construct, improvement in improvements.items():
            if improvement > 0:
                cost    = improvement * cost_per_unit[construct]
                tri_gain = gains[construct]
                roi[construct] = {
                    'Improvement': improvement,
                    'Est. Cost (KSh)': int(cost),
                    'TRI Gain': round(tri_gain, 3),
                    'TRI Gain per 10k KSh': round(tri_gain / (cost/10000), 3) if cost > 0 else 0
                }
        if roi:
            roi_df = pd.DataFrame(roi).T.reset_index()
            roi_df.columns = ['Construct','Improvement Needed',
                               'Est. Cost (KSh)','Expected TRI Gain',
                               'TRI Gain per KSh 10k']
            roi_df = roi_df.sort_values('TRI Gain per KSh 10k', ascending=False)
            st.dataframe(roi_df, width='stretch')

            best = roi_df.iloc[0]
            total_cost = roi_df['Est. Cost (KSh)'].sum()
            if total_cost > budget:
                st.warning(
                    f"Total estimated cost (KSh {total_cost:,}) exceeds budget "
                    f"(KSh {budget:,}). Highest-return intervention: "
                    f"**{best['Construct']}** "
                    f"(est. KSh {int(best['Est. Cost (KSh)']):,})."
                )
            else:
                st.success(
                    f"Budget is sufficient. Highest-return intervention: "
                    f"**{best['Construct']}** — "
                    f"{best['TRI Gain per KSh 10k']:.2f} TRI points per KSh 10,000."
                )

# ============================================================
# TAB 3 — SCHOOL DASHBOARD (PRINCIPAL)
# ============================================================
with tab3:
    st.markdown("## 🏫 School Dashboard — Principal View")
    st.markdown(
        "Upload your school's teacher survey responses to generate "
        "a school-level readiness report, identify at-risk teachers, "
        "and produce an evidence-based school action plan."
    )

    st.markdown(
        '<div class="policy-box">'
        '<strong>👤 Designed for: School Principals</strong><br>'
        'This dashboard enables evidence-based professional development '
        'planning at the school level. All individual data is anonymised '
        '— teachers are identified by number only.'
        '</div>',
        unsafe_allow_html=True
    )

    # Template download
    with st.expander("📋 Download School Template"):
        school_template = pd.DataFrame({
            'teacher_id': [f'T{i:03d}' for i in range(1, 6)],
            'subject': ['Mathematics','English','Science','History','Arts'],
            'conf_B2': [4,3,2,4,3], 'conf_B6': [4,3,2,3,4],
            'digi_C1c': [3,4,2,3,4], 'digi_C1d': [3,3,2,4,3],
            'digi_C2e': [4,3,2,3,4],
            'res_D1': [3,4,2,3,3], 'res_D2': [3,3,2,4,3],
            'res_D3': [4,3,2,3,4], 'res_D4': [3,3,2,3,3],
            'res_D5': [3,4,2,3,3], 'res_D6': [3,3,2,4,3],
            'train_E2': [3,4,2,3,4], 'train_E3': [3,3,2,4,3],
            'train_E4': [4,3,2,3,4], 'train_E5': [3,4,2,3,3],
            'train_E6': [3,3,2,4,3], 'train_E7': [4,3,2,3,4],
            'training_attended': [1,1,0,1,1]
        })
        st.markdown(
            generate_csv_download(school_template, "school_template.csv"),
            unsafe_allow_html=True
        )

    school_file = st.file_uploader(
        "Upload School Teacher Data (CSV)", type=['csv'], key="school_upload"
    )

    if school_file:
        try:
            sdf = pd.read_csv(school_file)
            st.success(f"✅ Loaded {len(sdf)} teacher records")

            # Compute construct means and TRI
            ped_cols   = [c for c in sdf.columns if c in ['conf_B2','conf_B6']]
            dig_cols   = [c for c in sdf.columns if c in ['digi_C1c','digi_C1d','digi_C2e']]
            res_cols   = [c for c in sdf.columns if c.startswith('res_D')]
            train_cols = [c for c in sdf.columns if c.startswith('train_E')]

            for col_list, name, default in [
                (ped_cols,'Pedagogical_Confidence',2.5),
                (dig_cols,'Digital_Literacy',2.5),
                (res_cols,'Resource_Availability',2.5),
                (train_cols,'Training_Quality',2.5)
            ]:
                sdf[name] = sdf[col_list].mean(axis=1) if col_list else default

            sdf['TRI'] = (sdf['Pedagogical_Confidence'] +
                          sdf['Digital_Literacy'] +
                          sdf['Resource_Availability'] +
                          sdf['Training_Quality'])

            sdf['Readiness_Band'] = pd.cut(
                sdf['TRI'],
                bins=[0,9.99,14.99,20],
                labels=['Low','Moderate','High']
            )

            # Mismatch detection per teacher
            def row_mismatch(row):
                skill = (row['Digital_Literacy'] +
                         row['Resource_Availability'] +
                         row['Training_Quality']) / 3
                gap = row['Pedagogical_Confidence'] - skill
                if gap > 1.2:   return 'Overconfident'
                elif gap < -1.2: return 'Underconfident'
                else:            return 'Aligned'

            sdf['Profile_Type'] = sdf.apply(row_mismatch, axis=1)

            # School summary
            st.markdown("### 📊 School Readiness Summary")
            m1,m2,m3,m4 = st.columns(4)
            with m1:
                st.metric("School Mean TRI", f"{sdf['TRI'].mean():.2f}")
            with m2:
                n_low = (sdf['Readiness_Band']=='Low').sum()
                st.metric("At-Risk Teachers (Low)", n_low,
                          delta_color="inverse" if n_low > 0 else "off")
            with m3:
                n_mis = (sdf['Profile_Type']!='Aligned').sum()
                st.metric("Mismatch Profiles", n_mis)
            with m4:
                below_thresh = (sdf['Training_Quality'] < 3.0).sum()
                st.metric("Below Training Threshold", below_thresh)

            # Band distribution
            st.markdown("### 🎯 Readiness Band Distribution")
            c1, c2 = st.columns(2)
            band_counts = sdf['Readiness_Band'].value_counts().reset_index()
            band_counts.columns = ['Band','Count']
            with c1:
                fig = px.pie(
                    band_counts, values='Count', names='Band',
                    color='Band',
                    color_discrete_map={
                        'Low':'#dc3545','Moderate':'#ffc107','High':'#28a745'
                    },
                    title='Readiness Band Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.bar(
                    band_counts, x='Band', y='Count', color='Band',
                    color_discrete_map={
                        'Low':'#dc3545','Moderate':'#ffc107','High':'#28a745'
                    },
                    title='Teachers by Readiness Band'
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Staff Heat Map (anonymised)
            st.markdown("### 🌡️ Staff Readiness Heat Map (Anonymised)")
            heat_df = sdf[['Pedagogical_Confidence','Digital_Literacy',
                           'Resource_Availability','Training_Quality']].copy()
            heat_df.index = [f'Teacher {i+1}' for i in range(len(heat_df))]
            fig_heat = px.imshow(
                heat_df,
                color_continuous_scale='RdYlGn',
                zmin=1, zmax=5,
                title='Construct Scores by Teacher (Anonymised)',
                labels={'color':'Score (1–5)'}
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Department breakdown if available
            if 'subject' in sdf.columns:
                st.markdown("### 📚 Readiness by Subject / Department")
                dept_df = sdf.groupby('subject')[
                    ['Pedagogical_Confidence','Digital_Literacy',
                     'Resource_Availability','Training_Quality','TRI']
                ].mean().reset_index()
                fig_dept = px.bar(
                    dept_df.melt(
                        id_vars='subject',
                        value_vars=['Pedagogical_Confidence',
                                    'Digital_Literacy',
                                    'Resource_Availability',
                                    'Training_Quality'],
                        var_name='Construct', value_name='Mean Score'
                    ),
                    x='subject', y='Mean Score', color='Construct',
                    barmode='group',
                    title='Average Construct Scores by Department',
                    range_y=[1,5]
                )
                st.plotly_chart(fig_dept, use_container_width=True)

            # Peer Mentoring Suggestions
            st.markdown("### 🤝 Peer Mentoring Recommendations")
            high_teachers = sdf[sdf['Readiness_Band']=='High'].index.tolist()
            low_teachers  = sdf[sdf['Readiness_Band']=='Low'].index.tolist()
            if high_teachers and low_teachers:
                pairs = []
                for i, low in enumerate(low_teachers):
                    mentor = high_teachers[i % len(high_teachers)]
                    pairs.append({
                        'Mentee (Low Readiness)': f'Teacher {low+1}',
                        'Mentor (High Readiness)': f'Teacher {mentor+1}',
                        'Focus Area': sdf.loc[low,
                            ['Pedagogical_Confidence','Digital_Literacy',
                             'Resource_Availability','Training_Quality']
                        ].idxmin()
                    })
                st.dataframe(pd.DataFrame(pairs), width='stretch')
            else:
                st.info("Mentoring pairs require at least one High and one "
                        "Low readiness teacher in the cohort.")

            # School Action Plan
            st.markdown("### 📋 Auto-Generated School Action Plan")
            weakest_construct = sdf[
                ['Pedagogical_Confidence','Digital_Literacy',
                 'Resource_Availability','Training_Quality']
            ].mean().idxmin().replace('_',' ')

            action_map = {
                'Pedagogical Confidence': (
                    "Organise structured peer observation cycles this term. "
                    "Arrange for at least 3 coaching sessions per at-risk teacher."
                ),
                'Digital Literacy': (
                    "Schedule a school-based digital tools workshop. "
                    "Ensure all teachers have access to devices and internet "
                    "connectivity during lesson preparation periods."
                ),
                'Resource Availability': (
                    "Submit a resource needs report to the county education office. "
                    "Explore parent-teacher association funding or NGO partnerships "
                    "for ICT procurement."
                ),
                'Training Quality': (
                    "Replace generic Ministry workshop attendance with school-contextualised "
                    "CBC professional development. Partner with a Teacher Training College "
                    "for subject-specific follow-up coaching."
                )
            }

            st.markdown(
                f'<div class="recommendation-box">'
                f'<strong>🏆 School Priority: {weakest_construct}</strong><br><br>'
                f'{action_map.get(weakest_construct, "Contact your county education officer for guidance.")}'
                f'</div>',
                unsafe_allow_html=True
            )

            # Export
            st.markdown("### 📥 Export School Report")
            export_df = sdf[['Pedagogical_Confidence','Digital_Literacy',
                              'Resource_Availability','Training_Quality',
                              'TRI','Readiness_Band','Profile_Type']].copy()
            export_df.index = [f'Teacher {i+1}' for i in range(len(export_df))]
            st.markdown(
                generate_csv_download(export_df.reset_index(),
                                      "school_readiness_report.csv"),
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ============================================================
# TAB 4 — COUNTY DASHBOARD
# ============================================================
with tab4:
    st.markdown("## 🏛️ County Dashboard — County Education Officer View")
    st.markdown(
        "Upload multi-school data to compare readiness across schools, "
        "identify county-wide resource gaps, and generate a professional "
        "development allocation plan for submission to the Ministry."
    )

    st.markdown(
        '<div class="policy-box">'
        '<strong>👤 Designed for: County Education Officers</strong><br>'
        'Upload a combined CSV with data from multiple schools. '
        'Include a <em>school_name</em> or <em>school_id</em> column '
        'to enable school-level comparison.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.expander("📋 Download County Template"):
        county_template = pd.DataFrame({
            'school_id': ['SCH001']*3 + ['SCH002']*3,
            'school_type': ['Public County']*3 + ['Private']*3,
            'location': ['Urban','Urban','Rural','Peri-urban','Urban','Rural'],
            'conf_B2': [4,3,2,4,3,2], 'conf_B6': [4,3,2,3,4,2],
            'digi_C1c': [3,4,2,3,4,1], 'digi_C1d': [3,3,2,4,3,2],
            'digi_C2e': [4,3,2,3,4,2],
            'res_D1': [3,4,2,3,3,1], 'res_D2': [3,3,2,4,3,2],
            'res_D3': [4,3,2,3,4,1], 'res_D4': [3,3,2,3,3,2],
            'res_D5': [3,4,2,3,3,1], 'res_D6': [3,3,2,4,3,2],
            'train_E2': [3,4,2,3,4,1], 'train_E3': [3,3,2,4,3,2],
            'train_E4': [4,3,2,3,4,1], 'train_E5': [3,4,2,3,3,2],
            'train_E6': [3,3,2,4,3,1], 'train_E7': [4,3,2,3,4,2],
            'training_attended': [1,1,0,1,1,0]
        })
        st.markdown(
            generate_csv_download(county_template, "county_template.csv"),
            unsafe_allow_html=True
        )

    county_file = st.file_uploader(
        "Upload County-Level Teacher Data (CSV)", type=['csv'],
        key="county_upload"
    )

    if county_file:
        try:
            cdf = pd.read_csv(county_file)
            st.success(f"✅ Loaded {len(cdf)} teacher records")

            # Compute constructs
            for cols, name in [
                ([c for c in cdf.columns if c in ['conf_B2','conf_B6']],
                 'Pedagogical_Confidence'),
                ([c for c in cdf.columns if c in ['digi_C1c','digi_C1d','digi_C2e']],
                 'Digital_Literacy'),
                ([c for c in cdf.columns if c.startswith('res_D')],
                 'Resource_Availability'),
                ([c for c in cdf.columns if c.startswith('train_E')],
                 'Training_Quality'),
            ]:
                cdf[name] = cdf[cols].mean(axis=1) if cols else 2.5

            cdf['TRI'] = (cdf['Pedagogical_Confidence'] +
                          cdf['Digital_Literacy'] +
                          cdf['Resource_Availability'] +
                          cdf['Training_Quality'])

            cdf['Readiness_Band'] = pd.cut(
                cdf['TRI'], bins=[0,9.99,14.99,20],
                labels=['Low','Moderate','High']
            )

            # County summary metrics
            st.markdown("### 📊 County Readiness Summary")
            m1,m2,m3,m4 = st.columns(4)
            with m1:
                st.metric("County Mean TRI", f"{cdf['TRI'].mean():.2f}")
            with m2:
                pct_low = (cdf['Readiness_Band']=='Low').mean() * 100
                st.metric("% Low Readiness", f"{pct_low:.1f}%")
            with m3:
                below_tq = (cdf['Training_Quality'] < 3.0).mean() * 100
                st.metric("% Below TQ Threshold", f"{below_tq:.1f}%")
            with m4:
                n_schools = cdf['school_id'].nunique() if 'school_id' in cdf.columns else 'N/A'
                st.metric("Schools Represented", n_schools)

            # Multi-school comparison
            if 'school_id' in cdf.columns:
                st.markdown("### 🏫 School-Level Comparison")
                school_summary = cdf.groupby('school_id').agg(
                    Mean_TRI=('TRI','mean'),
                    N_Teachers=('TRI','count'),
                    Pct_Low=('Readiness_Band', lambda x: (x=='Low').mean()*100),
                    Pedagogical=('Pedagogical_Confidence','mean'),
                    Digital=('Digital_Literacy','mean'),
                    Resources=('Resource_Availability','mean'),
                    Training=('Training_Quality','mean')
                ).reset_index().round(2)

                school_summary = school_summary.sort_values('Mean_TRI')

                fig = px.bar(
                    school_summary, x='school_id', y='Mean_TRI',
                    color='Mean_TRI',
                    color_continuous_scale='RdYlGn',
                    title='Mean TRI by School (sorted lowest to highest)',
                    labels={'school_id':'School','Mean_TRI':'Mean TRI Score'},
                    text='Mean_TRI'
                )
                fig.add_hline(y=11.46, line_dash='dash', line_color='grey',
                              annotation_text='County Norm: 11.46')
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(school_summary, width='stretch')

            # Location comparison
            if 'location' in cdf.columns:
                st.markdown("### 🌍 Readiness by Location")
                loc_df = cdf.groupby('location')[
                    ['Pedagogical_Confidence','Digital_Literacy',
                     'Resource_Availability','Training_Quality','TRI']
                ].mean().reset_index().round(2)

                fig_loc = px.bar(
                    loc_df.melt(id_vars='location',
                                value_vars=['Pedagogical_Confidence',
                                            'Digital_Literacy',
                                            'Resource_Availability',
                                            'Training_Quality'],
                                var_name='Construct', value_name='Mean Score'),
                    x='location', y='Mean Score', color='Construct',
                    barmode='group',
                    title='Mean Construct Scores by Location',
                    range_y=[1,5]
                )
                fig_loc.add_hline(y=3.0, line_dash='dot', line_color='red',
                                  annotation_text='Moderate Threshold')
                st.plotly_chart(fig_loc, use_container_width=True)

            # PD Gap Analysis
            st.markdown("### 📚 Professional Development Gap Analysis")
            construct_means = cdf[['Pedagogical_Confidence','Digital_Literacy',
                                   'Resource_Availability','Training_Quality']].mean()
            gap_df = pd.DataFrame({
                'Construct': construct_means.index,
                'County Mean': construct_means.values,
                'Gap to Threshold (3.0)': np.maximum(3.0 - construct_means.values, 0)
            }).round(3)

            fig_gap = px.bar(
                gap_df, x='Construct', y='Gap to Threshold (3.0)',
                color='Gap to Threshold (3.0)',
                color_continuous_scale='Reds',
                title='Professional Development Gap by Construct '
                      '(higher bar = greater need)',
                text=gap_df['Gap to Threshold (3.0)'].round(2)
            )
            st.plotly_chart(fig_gap, use_container_width=True)

            # Resource Allocation Priority
            st.markdown("### 💰 Resource Allocation Priority Ranking")
            if 'school_id' in cdf.columns:
                priority = cdf.groupby('school_id').agg(
                    Mean_TRI=('TRI','mean'),
                    N_Teachers=('TRI','count'),
                    Pct_Low=('Readiness_Band', lambda x: (x=='Low').mean()*100),
                ).reset_index().sort_values('Mean_TRI')
                priority['Priority_Rank'] = range(1, len(priority)+1)
                priority['Recommendation'] = priority['Mean_TRI'].apply(
                    lambda s: '🔴 Urgent Support Needed' if s < 9
                    else ('🟡 Targeted Support' if s < 12 else '🟢 Maintenance'))
                st.dataframe(priority.round(2), width='stretch')

            # Training Quality Coverage
            if 'training_attended' in cdf.columns:
                st.markdown("### 📋 Training Coverage vs Quality")
                att_rate = cdf['training_attended'].mean() * 100
                quality_among_attended = cdf[cdf['training_attended']==1]['Training_Quality'].mean()
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Training Attendance Rate", f"{att_rate:.1f}%")
                with c2:
                    st.metric("Avg Quality (Attendees Only)",
                              f"{quality_among_attended:.2f}/5")
                with c3:
                    trained_low_quality = (
                        (cdf['training_attended']==1) &
                        (cdf['Training_Quality'] < 3.0)
                    ).sum()
                    st.metric("Trained but Low Quality", trained_low_quality,
                              help="Teachers who attended but rated quality below 3.0")

            # County Report Export
            st.markdown("### 📥 Download County Report")
            county_report = cdf[
                ['Pedagogical_Confidence','Digital_Literacy',
                 'Resource_Availability','Training_Quality',
                 'TRI','Readiness_Band']
                + (['school_id'] if 'school_id' in cdf.columns else [])
                + (['location'] if 'location' in cdf.columns else [])
            ]
            st.markdown(
                generate_csv_download(county_report,
                                      "county_readiness_report.csv"),
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error processing county file: {e}")

# ============================================================
# TAB 5 — POLICY INTELLIGENCE
# ============================================================
with tab5:
    st.markdown("## 📋 Policy Intelligence — Ministry of Education View")
    st.markdown(
        "Strategic readiness intelligence to support evidence-based policy "
        "for Kenya's 2026 senior secondary CBC rollout."
    )

    st.markdown(
        '<div class="policy-box">'
        '<strong>👤 Designed for: Ministry of Education Officials, '
        'Policymakers, and Researchers</strong><br>'
        'This module provides national-level diagnostic insights, '
        'rollout readiness indicators, equity monitoring, and '
        'alignment with Sustainable Development Goal 4.'
        '</div>',
        unsafe_allow_html=True
    )

    # Reference Benchmarks
    st.markdown("### 📐 National Reference Benchmarks (Nakuru County Norms)")
    st.caption(
        "These norms are derived from 1,200 teachers across 240 secondary "
        "schools in Nakuru County (January 2025). Use these as reference "
        "benchmarks when comparing data from other counties."
    )
    bench_df = pd.DataFrame({
        'Construct': ['Pedagogical Confidence','Digital Literacy',
                      'Resource Availability','Training Quality','TRI (4-20 scale)'],
        'County Mean': ['3.19', '2.66', '2.85', '2.76', '11.46'],
        'Std Deviation': ['0.88', '0.77', '0.87', '1.20', '1.85'],
        'Low Band Threshold': ['under 2.5','under 2.0','under 2.0','under 1.5','under 10.0'],
        'Moderate Band': ['2.5 to 3.5','2.0 to 3.5','2.0 to 3.5','1.5 to 3.5','10.0 to 14.9'],
        'High Band': ['above 3.5','above 3.5','above 3.5','above 3.5','15.0 or above'],
    })
    st.dataframe(bench_df, width='stretch')

    st.markdown("---")

    # 2026 Rollout Readiness — Upload Section
    st.markdown("### 🚦 2026 Senior Secondary CBC Rollout Readiness Indicator")
    st.caption(
        "Upload county or school-level data to generate a traffic-light "
        "readiness indicator for the 2026 senior secondary rollout."
    )

    policy_file = st.file_uploader(
        "Upload Data for Rollout Assessment (CSV)", type=['csv'],
        key="policy_upload"
    )

    if policy_file:
        try:
            pdf = pd.read_csv(policy_file)

            for cols, name in [
                ([c for c in pdf.columns if c in ['conf_B2','conf_B6']],
                 'Pedagogical_Confidence'),
                ([c for c in pdf.columns if c in ['digi_C1c','digi_C1d','digi_C2e']],
                 'Digital_Literacy'),
                ([c for c in pdf.columns if c.startswith('res_D')],
                 'Resource_Availability'),
                ([c for c in pdf.columns if c.startswith('train_E')],
                 'Training_Quality'),
            ]:
                pdf[name] = pdf[cols].mean(axis=1) if cols else 2.5

            pdf['TRI'] = (pdf['Pedagogical_Confidence'] +
                          pdf['Digital_Literacy'] +
                          pdf['Resource_Availability'] +
                          pdf['Training_Quality'])

            mean_tri  = pdf['TRI'].mean()
            pct_low   = (pdf['TRI'] < 10).mean() * 100
            pct_high  = (pdf['TRI'] >= 15).mean() * 100
            tq_pct    = (pdf['Training_Quality'] >= 3.0).mean() * 100

            # Traffic light determination
            if mean_tri >= 13 and pct_low < 10:
                light, light_class, light_msg = (
                    "🟢 ON TRACK", "traffic-green",
                    "The uploaded cohort meets the readiness benchmarks "
                    "for the 2026 senior secondary CBC rollout."
                )
            elif mean_tri >= 11 and pct_low < 20:
                light, light_class, light_msg = (
                    "🟡 MODERATE RISK", "traffic-amber",
                    "The uploaded cohort shows moderate readiness. "
                    "Targeted interventions are recommended before 2026."
                )
            else:
                light, light_class, light_msg = (
                    "🔴 HIGH RISK", "traffic-red",
                    "The uploaded cohort shows significant readiness gaps. "
                    "Urgent, system-level professional development is required "
                    "before the 2026 rollout."
                )

            c1, c2 = st.columns([1,2])
            with c1:
                st.markdown(
                    f'<div class="{light_class}" '
                    f'style="margin-top:1rem; padding:1.5rem; text-align:center; '
                    f'border-radius:15px; font-size:1.5rem">{light}</div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f'<div class="policy-box" style="margin-top:1rem">'
                    f'{light_msg}<br><br>'
                    f'<strong>Key metrics:</strong> Mean TRI = {mean_tri:.2f} | '
                    f'Low Readiness = {pct_low:.1f}% | '
                    f'High Readiness = {pct_high:.1f}% | '
                    f'Training Quality ≥ 3.0: {tq_pct:.1f}% of teachers'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")

        except Exception as e:
            st.error(f"Error: {e}")

    # Policy Sequencing Recommendation — always visible
    st.markdown("### 📐 Evidence-Based Policy Sequencing Recommendation")
    st.markdown(
        '<div class="recommendation-box">'
        '<strong>Research Finding — Digital Literacy × Training Quality Threshold</strong><br><br>'
        'Analysis of 1,200 teachers using SHAP interaction analysis identified a '
        'critical conditional threshold: <strong>Digital Literacy investments yield '
        'minimal readiness gains when Training Quality is below 3.0 on the 5-point scale.</strong>'
        '<br><br>'
        '<strong>Policy Implication:</strong> Professional development sequencing should follow '
        'this evidence-based order:<br>'
        '1️⃣ <strong>Phase 1:</strong> Improve Training Quality to ≥ 3.0 '
        '(contextualised, school-based, subject-specific PD)<br>'
        '2️⃣ <strong>Phase 2:</strong> Build Pedagogical Confidence '
        '(mastery experiences, peer observation, coaching)<br>'
        '3️⃣ <strong>Phase 3:</strong> Digital skills upskilling '
        '(only effective once Phase 1 is achieved)<br>'
        '4️⃣ <strong>Phase 4:</strong> Resource infrastructure investment '
        '(sustains and amplifies gains from Phases 1–3)'
        '</div>',
        unsafe_allow_html=True
    )

    # Reform Risk Dashboard
    st.markdown("### ⚠️ Reform Risk Indicators")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="danger-box">'
            '<strong>🔴 High-Risk Indicator Thresholds</strong><br><br>'
            '• Any county where > 20% of teachers score in the Low band<br>'
            '• Any school where Training Quality mean < 2.0<br>'
            '• Rural schools where Digital Literacy mean < 2.0 '
            '(urban-rural gap > 0.8 points documented in Nakuru data)<br>'
            '• Schools where ≥ 11% of teachers exhibit '
            'confidence-skills mismatch profiles'
            '</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="success-box">'
            '<strong>🟢 On-Track Indicator Thresholds</strong><br><br>'
            '• County mean TRI ≥ 13.0 (above Nakuru county mean + 0.5 SD)<br>'
            '• Training Quality mean ≥ 3.0 for at least 70% of teachers<br>'
            '• Digital Literacy mean ≥ 3.0 in both urban and rural schools<br>'
            '• Fewer than 5% of teachers in the Low readiness band'
            '</div>',
            unsafe_allow_html=True
        )

    # Equity Monitor
    st.markdown("### ⚖️ Equity Monitor")
    st.markdown(
        '<div class="policy-box">'
        '<strong>Urban-Rural Digital Equity Gap (Nakuru County Reference)</strong><br><br>'
        'Measurement invariance analysis confirmed a mean Digital Literacy difference '
        'of <strong>0.819 scale points</strong> between urban and rural schools '
        'in Nakuru County — the largest equity gap identified across all four constructs. '
        'This gap reflects genuine infrastructure disparities, not measurement bias.<br><br>'
        '<strong>Policy Action Required:</strong> Rural school digital infrastructure '
        'investment is a prerequisite for equitable CBC implementation. '
        'County education officers should prioritise rural schools in resource allocation.'
        '</div>',
        unsafe_allow_html=True
    )

    equity_data = pd.DataFrame({
        'Construct': ['Pedagogical Confidence','Digital Literacy',
                      'Resource Availability','Training Quality'],
        'Urban Mean (Nakuru Ref.)': [3.20, 3.09, 2.85, 2.76],
        'Rural Mean (Nakuru Ref.)': [3.19, 2.27, 2.84, 2.77],
        'Urban-Rural Gap': [0.01, 0.82, 0.01, 0.00],
        'Equity Concern': ['None','High','Minimal','None']
    })
    st.dataframe(equity_data, width='stretch')

    fig_equity = px.bar(
        equity_data.melt(id_vars='Construct',
                         value_vars=['Urban Mean (Nakuru Ref.)',
                                     'Rural Mean (Nakuru Ref.)'],
                         var_name='Location', value_name='Mean Score'),
        x='Construct', y='Mean Score', color='Location',
        barmode='group',
        title='Urban vs Rural Mean Construct Scores (Nakuru County Reference Data)',
        range_y=[1,5],
        color_discrete_map={
            'Urban Mean (Nakuru Ref.)': '#2a5298',
            'Rural Mean (Nakuru Ref.)': '#e67e22'
        }
    )
    fig_equity.add_hline(y=3.0, line_dash='dash', line_color='red',
                         annotation_text='Moderate Threshold (3.0)')
    st.plotly_chart(fig_equity, use_container_width=True)

    # Training Attendance vs Quality Policy
    st.markdown("### 📊 Training Quality vs Attendance Policy Finding")
    st.markdown(
        '<div class="recommendation-box">'
        '<strong>Empirical Finding for Ministry Monitoring Frameworks</strong><br><br>'
        'The binary training attendance variable was <strong>non-significant</strong> '
        'across all predictive models (p = 0.364), while all six Training Quality items '
        'were highly significant (β = 0.141–0.207, p < 0.001).<br><br>'
        '<strong>Policy Implication:</strong> Current Ministry monitoring frameworks '
        'that track attendance counts as a proxy for professional development impact '
        'are measuring a variable that does not predict readiness outcomes. '
        'The six Training Quality items from the TRI instrument '
        '(train_E2–E7) are recommended as a standardised post-training '
        'evaluation tool to replace attendance registers.'
        '</div>',
        unsafe_allow_html=True
    )

    att_qual_df = pd.DataFrame({
        'Metric': ['Training Attendance (Binary)',
                   'train_E2: Comprehensiveness',
                   'train_E3: Practical Applicability',
                   'train_E4: Trainer Effectiveness',
                   'train_E5: Practice Opportunities',
                   'train_E6: Follow-up Support',
                   'train_E7: Confidence Improvement'],
        'MLR Coefficient (beta)': ['0.033', '0.167', '0.164', '0.161',
                                    '0.207', '0.141', '0.157'],
        'p-value': ['0.364', 'less than 0.001', 'less than 0.001',
                    'less than 0.001', 'less than 0.001',
                    'less than 0.001', 'less than 0.001'],
        'Significant': ['No','Yes','Yes','Yes','Yes','Yes','Yes']
    })
    st.dataframe(att_qual_df, width='stretch')

    # SDG 4 Alignment
    st.markdown("### 🌍 SDG 4 Alignment Tracker")
    st.markdown(
        '<span class="sdg-badge">SDG 4 — Quality Education</span>',
        unsafe_allow_html=True
    )
    st.markdown("")
    sdg_df = pd.DataFrame({
        'SDG 4 Target': [
            '4.1 — Ensure all learners complete quality primary and secondary education',
            '4.4 — Increase ICT-skilled youth and adults',
            '4.c — Increase supply of qualified teachers'
        ],
        'TRI Contribution': [
            'Teachers in the High readiness band are more likely to deliver '
            'quality, competency-based instruction aligned with Target 4.1',
            'Digital Literacy construct directly measures teachers\' '
            'technology integration capability, a prerequisite for teaching '
            'Target 4.4 competencies',
            'The TRI provides a validated, scalable mechanism for monitoring '
            'teacher preparedness — essential for Target 4.c reporting'
        ],
        'TRI Metric': [
            'TRI ≥ 15.0 (High Readiness Band)',
            'Digital Literacy mean ≥ 3.0',
            'Proportion of teachers with TRI ≥ 11.46 (county norm)'
        ]
    })
    st.dataframe(sdg_df, width='stretch')

# ============================================================
# TAB 6 — COHORT ANALYSIS
# ============================================================
with tab6:
    st.markdown("## 📊 Cohort Analysis")
    st.markdown(
        "Upload a CSV of teacher survey responses for any cohort to generate "
        "readiness distributions, resource gap analysis, and infrastructure "
        "priority rankings."
    )

    with st.expander("📋 Download Cohort Template"):
        cohort_template = pd.DataFrame({
            'conf_B2': [4,3,2,4,3], 'conf_B6': [4,3,2,3,4],
            'digi_C1c': [3,4,2,3,4], 'digi_C1d': [3,3,2,4,3],
            'digi_C2e': [4,3,2,3,4],
            'res_D1': [3,4,2,3,3], 'res_D2': [3,3,2,4,3],
            'res_D3': [4,3,2,3,4], 'res_D4': [3,3,2,3,3],
            'res_D5': [3,4,2,3,3], 'res_D6': [3,3,2,4,3],
            'train_E2': [3,4,2,3,4], 'train_E3': [3,3,2,4,3],
            'train_E4': [4,3,2,3,4], 'train_E5': [3,4,2,3,3],
            'train_E6': [3,3,2,4,3], 'train_E7': [4,3,2,3,4],
            'training_attended': [1,1,0,1,1]
        })
        st.markdown(
            generate_csv_download(cohort_template, "cohort_template.csv"),
            unsafe_allow_html=True
        )

    cohort_file = st.file_uploader(
        "Upload CSV with Teacher Responses", type=['csv'],
        key="cohort_upload"
    )

    if cohort_file:
        try:
            hdf = pd.read_csv(cohort_file)
            st.success(f"✅ Loaded {len(hdf)} teacher records")

            for cols, name in [
                ([c for c in hdf.columns if c in ['conf_B2','conf_B6']],
                 'Pedagogical_Confidence'),
                ([c for c in hdf.columns if c in ['digi_C1c','digi_C1d','digi_C2e']],
                 'Digital_Literacy'),
                ([c for c in hdf.columns if c.startswith('res_D')],
                 'Resource_Availability'),
                ([c for c in hdf.columns if c.startswith('train_E')],
                 'Training_Quality'),
            ]:
                hdf[name] = hdf[cols].mean(axis=1) if cols else 2.5

            hdf['TRI'] = (hdf['Pedagogical_Confidence'] +
                          hdf['Digital_Literacy'] +
                          hdf['Resource_Availability'] +
                          hdf['Training_Quality'])

            hdf['Readiness_Band'] = pd.cut(
                hdf['TRI'], bins=[0,9.99,14.99,20],
                labels=['Low','Moderate','High']
            )

            # Summary
            st.markdown("### 📊 Cohort Summary")
            m1,m2,m3,m4 = st.columns(4)
            with m1: st.metric("Mean TRI", f"{hdf['TRI'].mean():.2f}")
            with m2: st.metric("Median TRI", f"{hdf['TRI'].median():.2f}")
            with m3: st.metric("Min TRI", f"{hdf['TRI'].min():.2f}")
            with m4: st.metric("Max TRI", f"{hdf['TRI'].max():.2f}")

            # TRI histogram
            fig = px.histogram(
                hdf, x='TRI', nbins=20,
                title='TRI Score Distribution',
                labels={'TRI':'TRI Score (4–20)'},
                color_discrete_sequence=['#2a5298']
            )
            fig.add_vline(x=hdf['TRI'].mean(), line_dash='dash',
                          line_color='red',
                          annotation_text=f"Mean: {hdf['TRI'].mean():.2f}")
            fig.add_vline(x=11.46, line_dash='dot', line_color='grey',
                          annotation_text="County Norm: 11.46")
            st.plotly_chart(fig, use_container_width=True)

            # Band distribution
            c1, c2 = st.columns(2)
            band_c = hdf['Readiness_Band'].value_counts().reset_index()
            band_c.columns = ['Band','Count']
            with c1:
                fig = px.pie(
                    band_c, values='Count', names='Band',
                    color='Band',
                    color_discrete_map={
                        'Low':'#dc3545','Moderate':'#ffc107','High':'#28a745'
                    },
                    title='Readiness Band Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.bar(
                    band_c, x='Band', y='Count', color='Band',
                    color_discrete_map={
                        'Low':'#dc3545','Moderate':'#ffc107','High':'#28a745'
                    },
                    title='Teachers by Band'
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Construct averages
            st.markdown("### 🔧 Construct-Level Analysis")
            cons_df = pd.DataFrame({
                'Construct': ['Pedagogical Confidence','Digital Literacy',
                              'Resource Availability','Training Quality'],
                'Mean Score': [
                    hdf['Pedagogical_Confidence'].mean(),
                    hdf['Digital_Literacy'].mean(),
                    hdf['Resource_Availability'].mean(),
                    hdf['Training_Quality'].mean()
                ]
            })
            fig_c = px.bar(
                cons_df, x='Construct', y='Mean Score',
                color='Mean Score', color_continuous_scale='Viridis',
                title='Average Construct Scores', range_y=[1,5],
                text=cons_df['Mean Score'].round(2)
            )
            fig_c.add_hline(y=3.0, line_dash='dash', line_color='red',
                            annotation_text='Moderate Threshold (3.0)')
            st.plotly_chart(fig_c, use_container_width=True)

            # Resource Deficit Analysis
            st.markdown("### 🏗️ Resource Deficit Analysis")
            res_cols = [c for c in hdf.columns if c.startswith('res_D')]
            if res_cols:
                res_names = {
                    'res_D1':'ICT Infrastructure', 'res_D2':'Internet Connectivity',
                    'res_D3':'Learning Materials', 'res_D4':'Technical Support',
                    'res_D5':'Electricity Reliability', 'res_D6':'Equipment Maintenance'
                }
                res_means = hdf[res_cols].mean().rename(res_names)
                res_df = pd.DataFrame({
                    'Resource Item': res_means.index,
                    'Mean Score': res_means.values,
                    'Gap to Threshold': np.maximum(3.0 - res_means.values, 0)
                }).sort_values('Mean Score')

                fig_res = px.bar(
                    res_df, x='Resource Item', y='Mean Score',
                    color='Mean Score',
                    color_continuous_scale='RdYlGn',
                    title='Resource Availability by Item '
                          '(sorted lowest to highest)',
                    range_y=[1,5], text=res_df['Mean Score'].round(2)
                )
                fig_res.add_hline(y=3.0, line_dash='dash', line_color='red',
                                  annotation_text='Moderate Threshold')
                st.plotly_chart(fig_res, use_container_width=True)

                # Infrastructure Priority Ranker
                st.markdown("#### 🏆 Infrastructure Priority Ranker")
                st.caption(
                    "Items ranked by severity of deficit "
                    "(rank 1 = most urgent procurement priority)."
                )
                res_df['Priority Rank'] = range(1, len(res_df)+1)
                res_df['Action'] = res_df['Mean Score'].apply(
                    lambda s: '🔴 Urgent' if s < 2
                    else ('🟡 Important' if s < 3 else '🟢 Maintain'))
                st.dataframe(res_df.round(2), width='stretch')

            # Export
            st.markdown("### 📥 Export Cohort Analysis")
            export_h = hdf[['Pedagogical_Confidence','Digital_Literacy',
                             'Resource_Availability','Training_Quality',
                             'TRI','Readiness_Band']]
            st.markdown(
                generate_csv_download(export_h.reset_index(drop=True),
                                      "cohort_analysis.csv"),
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error processing cohort file: {e}")

# ============================================================
# TAB 7 — PROGRESS TRACKING
# ============================================================
with tab7:
    st.markdown("## 📈 Progress Tracking")
    st.markdown(
        "Track TRI scores across multiple assessments within this session "
        "to monitor change over time."
    )

    if st.session_state.previous_predictions:
        hist_df = pd.DataFrame(st.session_state.previous_predictions)

        st.markdown("### Assessment History")
        st.dataframe(hist_df, width='stretch')

        # Trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(hist_df)+1)),
            y=hist_df['TRI_Score'],
            mode='lines+markers',
            name='TRI Score',
            line=dict(color='#2a5298', width=2),
            marker=dict(size=8)
        ))
        fig.add_hline(y=11.46, line_dash='dash', line_color='grey',
                      annotation_text='County Norm: 11.46')
        fig.add_hrect(y0=15, y1=20, fillcolor='#28a745', opacity=0.1,
                      annotation_text='High', annotation_position='right')
        fig.add_hrect(y0=10, y1=15, fillcolor='#ffc107', opacity=0.1)
        fig.add_hrect(y0=4,  y1=10, fillcolor='#dc3545', opacity=0.1,
                      annotation_text='Low', annotation_position='right')
        fig.update_layout(
            title='TRI Score Trend Across Assessments',
            xaxis_title='Assessment Number',
            yaxis_title='TRI Score (4–20)',
            yaxis_range=[4, 20],
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Construct trends if available
        if all(c in hist_df.columns for c in
               ['Ped_Confidence','Digital_Literacy',
                'Resource_Availability','Training_Quality']):
            fig2 = go.Figure()
            for col, colour in [('Ped_Confidence','#e74c3c'),
                                 ('Digital_Literacy','#3498db'),
                                 ('Resource_Availability','#2ecc71'),
                                 ('Training_Quality','#9b59b6')]:
                if col in hist_df.columns:
                    fig2.add_trace(go.Scatter(
                        x=list(range(1, len(hist_df)+1)),
                        y=hist_df[col],
                        mode='lines+markers',
                        name=col.replace('_',' '),
                        line=dict(color=colour)
                    ))
            fig2.add_hline(y=3.0, line_dash='dot', line_color='grey',
                           annotation_text='Moderate Threshold')
            fig2.update_layout(
                title='Construct Score Trends',
                xaxis_title='Assessment Number',
                yaxis_title='Score (1–5)',
                yaxis_range=[1,5], height=380
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Summary stats
        st.markdown("### Summary Statistics")
        m1,m2,m3 = st.columns(3)
        with m1:
            st.metric("Total Assessments", len(hist_df))
        with m2:
            st.metric("Average TRI", f"{hist_df['TRI_Score'].mean():.1f}")
        with m3:
            if len(hist_df) > 1:
                change = hist_df['TRI_Score'].iloc[-1] - hist_df['TRI_Score'].iloc[0]
                st.metric("Score Change", f"{change:+.1f}")
            else:
                st.metric("Score Change", "N/A (need 2+ assessments)")

        st.markdown("### 📥 Export History")
        st.markdown(
            generate_csv_download(hist_df, "assessment_history.csv"),
            unsafe_allow_html=True
        )
    else:
        st.info(
            "No assessments recorded yet. "
            "Complete an assessment in the Individual Assessment tab."
        )

# ============================================================
# TAB 8 — TECHNICAL DOCUMENTATION
# ============================================================
with tab8:
    st.markdown("## ⚙️ Technical Documentation")

    with st.expander("📊 Model Performance", expanded=True):
        st.markdown("""
        **Final Stacking Ensemble — Held-Out Test Set Performance ($n = 97$ teachers):**
        - R² = **0.9763**
        - RMSE = **0.307**
        - MAE = **0.245** points
        - MAPE = **2.17%**
        - 87.6% of predictions within ±0.5 points
        - 100% of predictions within ±1.0 point

        **Model Architecture:**
        - Base Learners: MLR | Ridge Regression (λ=5.995) | XGBoost | SVR (sigmoid kernel)
        - Meta-Learner: Ridge Regression (λ=1.0)
        - Cross-Validation: 5-fold GroupKFold (school-grouped, zero leakage)
        """)

        weights_df = pd.DataFrame({
            'Base Model': ['SVR (sigmoid)','XGBoost','Ridge Regression','MLR'],
            'Meta-Learner Weight': [38.5, 27.7, 18.3, 16.5]
        })
        fig = px.pie(
            weights_df, values='Meta-Learner Weight', names='Base Model',
            title='Stacking Ensemble Meta-Learner Weights',
            color_discrete_sequence=['#9b59b6','#e67e22','#2ecc71','#3498db']
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔑 Feature Importance (MLR Coefficients)"):
        coef_df = pd.DataFrame({
            'Feature': ['conf_B2','conf_B6','digi_C1c','digi_C1d','digi_C2e',
                        'train_E5','res_D3','res_D5','train_E2','res_D1',
                        'res_D6','train_E3','train_E4','train_E7','res_D2',
                        'res_D4','train_E6','training_attended','Digital_x_Training'],
            'Coefficient': ['0.461','0.452','0.308','0.298','0.257',
                            '0.207','0.204','0.189','0.167','0.166',
                            '0.165','0.164','0.161','0.157','0.152',
                            '0.143','0.141','0.033','-0.002'],
            'Significant': ['Yes']*17 + ['No','No']
        })
        fig = px.bar(
            coef_df.head(10),
            x='Coefficient', y='Feature',
            orientation='h',
            color='Coefficient',
            color_continuous_scale='Blues',
            title='Top 10 MLR Predictors by Coefficient Magnitude'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(coef_df, width='stretch')

    with st.expander("📐 Psychometric Validation"):
        st.markdown("""
        **Exploratory Factor Analysis (n = 840 calibration subsample):**
        - KMO = 0.952 (marvelous adequacy)
        - Bartlett's test: χ² = 34,853.28, df = 378, p < 0.001
        - Parallel analysis: 4 factors retained (eigenvalue cliff: 4.562 → 0.423)
        - Total variance explained: 77.0%
        - All factor loadings: 0.803–0.938; no cross-loadings > 0.07

        **Confirmatory Factor Analysis (n = 360 holdout subsample):**
        - CFI = 0.996 | TLI = 0.996 | RMSEA = 0.017 | χ²/df = 1.11

        **Reliability:**
        """)
        rel_df = pd.DataFrame({
            'Construct': ['Pedagogical Confidence','Digital Literacy',
                          'Resource Availability','Training Quality'],
            'Items': [7,9,6,6],
            "Cronbach's α": [0.962, 0.952, 0.947, 0.976],
            "McDonald's ω": [0.962, 0.953, 0.947, 0.976],
            'Avg Inter-Item r': [0.781, 0.692, 0.750, 0.869]
        })
        st.dataframe(rel_df, width='stretch')

        st.markdown("""
        **Measurement Invariance:**
        - Full scalar invariance confirmed across gender (ΔCFI < 0.01)
        - Digital Literacy shows substantive urban-rural mean difference (0.819 points)
          — confirmed as genuine infrastructure gap, not measurement artefact
        """)

    with st.expander("🔑 Key Research Findings"):
        st.markdown("""
        1. **Pedagogical Confidence** is the strongest individual predictor (β = 0.461)
        2. **Training Quality** has the broadest distributional influence 
           (mean |SHAP| = 1.76)
        3. **Training attendance is non-significant** (p = 0.364) — 
           quality, not participation, determines impact
        4. **Digital Literacy × Training Quality threshold:** Digital skills 
           yield minimal TRI gains when Training Quality < 3.0
        5. **~11% of teachers** exhibit a confidence-skills mismatch profile
        6. **Linear relationships dominate** — MLR achieves R² = 0.9699; 
           tree-based models underperform
        7. **Urban-rural Digital Literacy gap = 0.819** scale points
        """)

    with st.expander("📚 References"):
        st.markdown("""
        - Bandura, A. (1997). *Self-efficacy: The exercise of control*. W.H. Freeman.
        - Chapman, P., et al. (2000). *CRISP-DM 1.0*. SPSS.
        - Desimone, L. M. (2009). Improving impact studies of teachers' professional development. *Educational Researcher, 38*(3), 181–199.
        - Fullan, M. (2015). *The new meaning of educational change* (5th ed.). Teachers College Press.
        - Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS, 30*.
        - Mishra, P., & Koehler, M. J. (2006). Technological pedagogical content knowledge. *Teachers College Record, 108*(6), 1017–1054.
        - Rogers, E. M. (2003). *Diffusion of innovations* (5th ed.). Free Press.
        - KICD. (2024). *CBC implementation review*. Kenya Institute of Curriculum Development.
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>© 2026 Strathmore University — Institute of Mathematical Sciences</p>
    <p>CBC Teacher Readiness Diagnostic | Version 3.0 — Policy Intelligence Edition</p>
    <p>Principal Researcher: Peter Wachugu Kanyuira &nbsp;|&nbsp; 
       Supervisor: Dr. John Olukuru</p>
    <p style="font-size:0.8rem; margin-top:0.5rem">
        Built on validated research: R² = 0.9763 | 1,200 teachers | 240 schools | 
        Nakuru County, Kenya
    </p>
</div>
""", unsafe_allow_html=True)
