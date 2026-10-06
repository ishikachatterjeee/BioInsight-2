import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================
# LOAD DATA
# ==========================

equipment = pd.read_csv("equipment.csv")
maintenance = pd.read_csv("maintenance_logs.csv")

# ==========================
# RISK SCORE
# ==========================

equipment["Risk_Score"] = (
    (100 - equipment["Health_Score"]) * 0.4
    + equipment["Failure_Count"] * 5
    + (30 - equipment["Calibration_Due_Days"].clip(upper=30)) * 0.5
)

def classify_risk(score):
    if score >= 45:
        return "High Risk"
    elif score >= 25:
        return "Medium Risk"
    else:
        return "Low Risk"

equipment["Risk_Level"] = equipment["Risk_Score"].apply(classify_risk)

def recommend_action(row):
    if row["Risk_Level"] == "High Risk":
        return "Immediate inspection and preventive maintenance"

    elif row["Calibration_Due_Days"] < 15:
        return "Schedule calibration"

    elif row["Failure_Count"] >= 4:
        return "Review failure history and inspect device"

    elif row["Risk_Level"] == "Medium Risk":
        return "Monitor closely"

    else:
        return "Normal operation"

equipment["Maintenance_Recommendation"] = equipment.apply(
    recommend_action,
    axis=1
)

# ==========================
# CONDITION CLASSIFICATION
# ==========================

def classify_device(score):

    if score >= 85:
        return "Healthy"

    elif score >= 70:
        return "Monitor"

    else:
        return "Critical"

equipment["Condition"] = equipment["Health_Score"].apply(
    classify_device
)
# ==========================
# ML TRAINING LABEL
# ==========================

equipment["Failure_Risk"] = (
    (equipment["Health_Score"] < 75) &
    (equipment["Failure_Count"] >= 4)
).astype(int)
# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="BioInsight",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp{
    background:#f8fafc;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:white;
    border-right:1px solid #e2e8f0;
}

/* Metric cards */
[data-testid="metric-container"]{
/* Metric cards */
[data-testid="metric-container"]{
    background:white;
    border-radius:20px;
    padding:18px;
    box-shadow:0px 4px 20px rgba(0,0,0,0.06);
    min-height:100px;
    border:1px solid #e2e8f0;
    transition:0.3s;
}

[data-testid="metric-container"]:hover{
    transform:translateY(-4px);
}

/* Tables */
[data-testid="stDataFrame"]{
    background:white;
    border-radius:20px;
}

/* Headers */
h1{
    color:#0f172a;
}

h2,h3{
    color:#1e293b;
}

</style>
""", unsafe_allow_html=True)
# ==========================
# HEADER
# ==========================

st.markdown("""
<div style="
padding:40px;
border-radius:25px;
background:linear-gradient(
135deg,
#3B82F6,
#8B5CF6
);
color:white;
">

<h1 style="color:white;">
BioInsight
</h1>

<p style="font-size:20px;">
Clinical Engineering Intelligence Platform
</p>

<p>
Monitor biomedical equipment health,
maintenance performance and operational risk.
</p>

</div>
""", unsafe_allow_html=True)
st.write("")
st.write("")

st.sidebar.title("BioInsight")

st.sidebar.markdown("""
### Clinical Engineering Intelligence Platform

Biomedical equipment monitoring,
failure analytics,
predictive maintenance,
and operational insights.
""")
# ==========================
# KPI CARDS
# ==========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Devices",
        len(equipment)
    )

with col2:
    st.metric(
        "Average Health Score",
        round(equipment["Health_Score"].mean(), 1)
    )

with col3:
    st.metric(
        "Devices in Maintenance",
        len(
            equipment[
                equipment["Status"] == "Maintenance"
            ]
        )
    )

with col4:
    st.metric(
        "Total Maintenance Records",
        len(maintenance)
    )

    
 # ==========================
# EXECUTIVE SUMMARY
# ==========================

critical_devices = equipment[
    equipment["Condition"] == "Critical"
]

due_devices = equipment[
    equipment["Calibration_Due_Days"] < 30
]

c1, c2, c3 = st.columns(3)

with c1:
    st.success("🟢 Equipment Availability: 96%")

with c2:
    st.warning(f"🟡 Calibration Due: {len(due_devices)} Devices")

with c3:
    st.error(f"🔴 Critical Devices: {len(critical_devices)}")

# ==========================
# DEPARTMENT DISTRIBUTION
# ==========================

st.divider()

st.subheader("🏢 Department-wise Equipment Distribution")

dept_counts = equipment["Department"].value_counts()

fig = px.pie(
    values=dept_counts.values,
    names=dept_counts.index,
    title="Department-wise Equipment Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# LOW HEALTH DEVICES
# ==========================

st.divider()

st.subheader("Devices Requiring Attention")
low_health = equipment.sort_values(
    by="Health_Score"
).head(5)

st.dataframe(low_health)

# ==========================
# HIGH RISK DEVICES
# ==========================

st.divider()

st.subheader("High Risk Equipment")
high_risk = equipment.sort_values(
    by="Risk_Score",
    ascending=False
).head(10)

st.dataframe(high_risk)

# ==========================
# CALIBRATION ALERTS
# ==========================

st.divider()

st.subheader("📅 Calibration Due Soon")

due_devices = equipment[
    equipment["Calibration_Due_Days"] < 30
]

st.dataframe(due_devices)

# ==========================
# CONDITION OVERVIEW
# ==========================

st.divider()

st.subheader("❤️ Equipment Condition Overview")

condition_counts = equipment["Condition"].value_counts()

fig = px.pie(
    values=condition_counts.values,
    names=condition_counts.index,
    title="Equipment Condition Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# EXECUTIVE ALERTS
# ==========================

st.divider()

st.subheader("🚑 Executive Alerts")

critical_devices = equipment[
    equipment["Condition"] == "Critical"
]

st.error(
    f"{len(critical_devices)} critical devices require immediate attention."
)

st.warning(
    f"{len(due_devices)} devices require calibration within 30 days."
)

# ==========================
# FAILURE ANALYTICS
# ==========================

st.divider()

st.subheader("🔧 Most Common Failures")

failure_counts = maintenance["Fault"].value_counts()

fig = px.bar(
    x=failure_counts.index,
    y=failure_counts.values,
    title="Most Common Failures",
    labels={
        "x": "Failure Type",
        "y": "Count"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# ==========================
# DOWNTIME ANALYTICS
# ==========================

st.divider()

st.subheader("⏱ Downtime by Fault")

downtime = maintenance.groupby(
    "Fault"
)["Downtime_Hours"].sum()

fig = px.bar(
    x=downtime.index,
    y=downtime.values,
    title="Downtime by Fault",
    labels={
        "x": "Fault",
        "y": "Downtime Hours"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# ==========================
# SEARCH EQUIPMENT
# ==========================

st.divider()

st.subheader("🔍 Search Equipment")

search_id = st.text_input(
    "Enter Equipment ID (Example: EQ001)"
)

if search_id:

    result = equipment[
        equipment["Equipment_ID"].str.contains(
            search_id,
            case=False,
            na=False
        )
    ]

    if len(result) > 0:
        st.dataframe(result)

    else:
        st.info("No equipment found.")

# ==========================
# MAINTENANCE RECORDS
# ==========================

st.divider()

st.subheader("Maintenance Activity")
st.dataframe(
    maintenance.head(20)
)

# ==========================
# EQUIPMENT INVENTORY
# ==========================

st.subheader("Equipment Inventory")

st.dataframe(equipment)

# ==========================
# TROUBLESHOOTING ASSISTANT
# ==========================

st.divider()

st.subheader("🧠 Biomedical Troubleshooting Assistant")

knowledge_base = {

    "Low Pressure Alarm": {
        "cause": "Circuit Leak",
        "action": "Inspect tubing and replace damaged tubing"
    },

    "Battery Failure": {
        "cause": "Battery Degradation",
        "action": "Replace Battery"
    },

    "Sensor Failure": {
        "cause": "Sensor Malfunction",
        "action": "Replace or Recalibrate Sensor"
    },

    "Flow Error": {
        "cause": "Occlusion in Tubing",
        "action": "Inspect and Clear Blockage"
    },

    "Calibration Drift": {
        "cause": "Calibration Expired",
        "action": "Perform Calibration"
    },

    "Power Supply Failure": {
        "cause": "Faulty Power Unit",
        "action": "Replace Power Supply"
    },

    "Communication Error": {
        "cause": "Network or Interface Failure",
        "action": "Check Connections and Restart System"
    }
}

selected_fault = st.selectbox(
    "Select a Fault",
    list(knowledge_base.keys())
)

st.success(
    f"Possible Cause: {knowledge_base[selected_fault]['cause']}"
)

st.info(
    f"Recommended Action: {knowledge_base[selected_fault]['action']}"
)
# ==========================
# BIOMEDICAL KNOWLEDGE GRAPH
# ==========================

st.divider()

st.subheader("🕸 Biomedical Failure Knowledge Graph")

graph_data = {

    "Low Pressure Alarm": {
        "equipment": "Ventilator",
        "cause": "Circuit Leak",
        "action": "Replace Tubing"
    },

    "Battery Failure": {
        "equipment": "Patient Monitor",
        "cause": "Battery Degradation",
        "action": "Replace Battery"
    },

    "Sensor Failure": {
        "equipment": "Patient Monitor",
        "cause": "Sensor Malfunction",
        "action": "Replace Sensor"
    },

    "Flow Error": {
        "equipment": "Infusion Pump",
        "cause": "Tubing Occlusion",
        "action": "Clear Blockage"
    },

    "Communication Error": {
        "equipment": "ECG Machine",
        "cause": "Network Failure",
        "action": "Restart System"
    }
}

selected_graph_fault = st.selectbox(
    "Select Fault for Graph",
    list(graph_data.keys())
)

equipment_name = graph_data[selected_graph_fault]["equipment"]
cause = graph_data[selected_graph_fault]["cause"]
action = graph_data[selected_graph_fault]["action"]

G = nx.DiGraph()

G.add_edge(equipment_name, selected_graph_fault)
G.add_edge(selected_graph_fault, cause)
G.add_edge(cause, action)

fig, ax = plt.subplots(figsize=(8, 5))

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3500,
    font_size=9,
    ax=ax
)

st.pyplot(fig)
# ==========================
# PREDICTIVE MAINTENANCE
# ==========================

st.divider()

st.subheader("🔮 Predictive Maintenance Dashboard")

risk_counts = equipment["Risk_Level"].value_counts()

fig = px.pie(
    values=risk_counts.values,
    names=risk_counts.index,
    title="Equipment Risk Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("🚨 Devices Requiring Priority Maintenance")

high_risk_devices = equipment[
    equipment["Risk_Level"] == "High Risk"
]

st.dataframe(
    high_risk_devices[
        [
            "Equipment_ID",
            "Equipment_Name",
            "Department",
            "Risk_Level",
            "Risk_Score",
            "Maintenance_Recommendation"
        ]
    ]
)

# ==========================
# EQUIPMENT PREDICTOR
# ==========================

st.divider()

st.subheader("🧠 Equipment Maintenance Decision Support")

device_id = st.selectbox(
    "Select Equipment",
    equipment["Equipment_ID"]
)

device = equipment[
    equipment["Equipment_ID"] == device_id
].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Health Score",
        int(device["Health_Score"])
    )

with col2:
    st.metric(
        "Failure Count",
        int(device["Failure_Count"])
    )

with col3:
    st.metric(
        "Risk Score",
        round(device["Risk_Score"], 1)
    )

if device["Risk_Level"] == "High Risk":

    st.error("🔴 HIGH RISK DEVICE")

elif device["Risk_Level"] == "Medium Risk":

    st.warning("🟡 MEDIUM RISK DEVICE")

else:

    st.success("🟢 LOW RISK DEVICE")

st.write(
    f"**Equipment:** {device['Equipment_Name']}"
)

st.write(
    f"**Department:** {device['Department']}"
)

st.write(
    f"**Risk Level:** {device['Risk_Level']}"
)

st.write(
    f"**Recommendation:** {device['Maintenance_Recommendation']}"
)
# ==========================
# MACHINE LEARNING MODEL
# ==========================

features = equipment[
    [
        "Health_Score",
        "Failure_Count",
        "Utilization_Hours",
        "Calibration_Due_Days"
    ]
]

target = equipment["Failure_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
st.divider()

st.subheader("🤖 Machine Learning Failure Prediction")

selected_device = st.selectbox(
    "Select Device for Prediction",
    equipment["Equipment_ID"],
    key="ml_device"
)

device = equipment[
    equipment["Equipment_ID"] == selected_device
].iloc[0]

input_data = [[
    device["Health_Score"],
    device["Failure_Count"],
    device["Utilization_Hours"],
    device["Calibration_Due_Days"]
]]

prediction = model.predict(input_data)[0]

probability = model.predict_proba(input_data)[0][1]

if prediction == 1:

    st.error(
        f"⚠ High Failure Risk ({probability*100:.1f}%)"
    )

else:

    st.success(
        f"✅ Low Failure Risk ({probability*100:.1f}%)"
    )

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Health Score",
        int(device["Health_Score"])
    )

    st.metric(
        "Failure Count",
        int(device["Failure_Count"])
    )

with col2:
    st.metric(
        "Utilization Hours",
        int(device["Utilization_Hours"])
    )

    st.metric(
        "Calibration Due Days",
        int(device["Calibration_Due_Days"])
    )
# ==========================
# FOOTER
# ==========================

st.divider()

st.markdown("""
---
<center>

BioInsight v1.0

Clinical Engineering Intelligence Platform

</center>
""", unsafe_allow_html=True)