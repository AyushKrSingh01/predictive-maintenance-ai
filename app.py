import streamlit as st
import pandas as pd
import joblib



st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)



model = joblib.load(
    "models/predictive_maintenance_model.pkl"
)

THRESHOLD = 0.45



st.title("⚙️ Predictive Maintenance AI")

st.markdown(
    """
    **Machine Failure Prediction using Random Forest**

    Enter the machine's operating parameters to estimate
    the probability of machine failure.
    """
)

st.divider()



st.subheader("🔧 Machine Parameters")

col1, col2, col3 = st.columns(3)

with col1:

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=295.0,
        max_value=305.0,
        value=300.0,
        step=0.1
    )

with col2:

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=305.0,
        max_value=315.0,
        value=310.0,
        step=0.1
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1500,
        step=10
    )

with col3:

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=80.0,
        value=40.0,
        step=0.5
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


st.divider()



if st.button(
    "🔍 Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    
    failure_probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = int(
        failure_probability >= THRESHOLD
    )



    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Failure Probability",
            f"{failure_probability * 100:.2f}%"
        )

    with result_col2:

        if prediction == 1:

            st.error(
                "⚠️ Potential Machine Failure Detected"
            )

        else:

            st.success(
                "✅ Machine Operating Normally"
            )


    st.progress(
        float(failure_probability)
    )

    st.caption(
        f"Classification threshold: {THRESHOLD:.2f}"
    )



    st.subheader("📋 Input Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.write(
            f"**Machine Type:** {machine_type}"
        )

        st.write(
            f"**Air Temperature:** {air_temperature:.1f} K"
        )

    with summary_col2:

        st.write(
            f"**Process Temperature:** "
            f"{process_temperature:.1f} K"
        )

        st.write(
            f"**Rotational Speed:** "
            f"{rotational_speed} rpm"
        )

    with summary_col3:

        st.write(
            f"**Torque:** {torque:.1f} Nm"
        )

        st.write(
            f"**Tool Wear:** {tool_wear} min"
        )



st.divider()

st.subheader("🤖 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Algorithm",
        "Random Forest"
    )

with col2:

    st.metric(
        "F1 Score",
        "69.01%"
    )

with col3:

    st.metric(
        "Recall",
        "72.06%"
    )

with col4:

    st.metric(
        "Threshold",
        "0.45"
    )


st.markdown(
    """
    ### Model Development

    The model was selected after comparing:

    - Logistic Regression
    - Decision Tree
    - Random Forest

    Because machine failure is an imbalanced classification
    problem, precision, recall and F1-score were considered
    alongside accuracy.

    Random Forest achieved the highest F1-score among the
    evaluated models. Threshold analysis showed that a
    threshold of **0.45** increased recall from **66.18%**
    to **72.06%** while slightly improving F1-score.

    **Important:** Feature importance indicates predictive
    contribution, not causal relationships.
    """
)


st.divider()

st.subheader("📈 Model Feature Importance")

st.markdown(
    """
    Feature importance shows how much each input feature
    contributed to the Random Forest's predictions.
    Higher values indicate greater predictive contribution.
    """
)

feature_importance = pd.DataFrame({
    "Feature": [
        "Torque [Nm]",
        "Rotational speed [rpm]",
        "Tool wear [min]",
        "Air temperature [K]",
        "Process temperature [K]",
        "Type"
    ],
    "Importance": [
        0.305896,
        0.297352,
        0.209390,
        0.100740,
        0.068782,
        0.017839
    ]
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=True
)

st.bar_chart(
    feature_importance.set_index("Feature")["Importance"]
)

st.caption(
    "Feature importance represents predictive contribution, "
    "not causal influence."
)