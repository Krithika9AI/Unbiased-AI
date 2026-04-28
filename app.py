import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="Unbiased AI", page_icon="⚖️", layout="wide")

st.title("⚖️ Unbiased AI")
st.subheader("Ensuring Fairness and Detecting Bias in Automated Decisions")
st.markdown("**Google Solution Challenge 2026**")

st.markdown("---")

if st.button("🚀 Load Example Dataset (Adult Income)", type="primary", use_container_width=True):
    with st.spinner("Loading dataset and training model..."):
        try:
            # Load dataset
            url = "https://raw.githubusercontent.com/pooja2512/Adult-Census-Income/master/adult.csv"
            df = pd.read_csv(url, na_values='?')
            df = df.dropna().reset_index(drop=True)
            
            # Target: 1 if income > 50K
            df['income'] = (df['income'] == '>50K').astype(int)
            
            st.success(f"✅ Dataset loaded successfully! ({df.shape[0]:,} records)")

            # Protected Attribute Selection
            sensitive_attr = st.selectbox(
                "Select Protected Attribute",
                options=["sex", "race"],
                format_func=lambda x: "Gender" if x == "sex" else x.title()
            )

            # Prepare data
            X = df.drop(['income'], axis=1)
            y = df['income']
            sensitive = df[sensitive_attr]   # uses actual column name

            # Encode categorical features
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

            # Train-Test Split
            X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
                X, y, sensitive, test_size=0.3, random_state=42, stratify=y
            )

            # Train Model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            # Group Statistics
            results = pd.DataFrame({
                'true': y_test,
                'pred': y_pred,
                'group': s_test
            })

            group_stats = results.groupby('group').agg(
                accuracy=('pred', lambda x: accuracy_score(results.loc[x.index, 'true'], x)),
                approval_rate=('pred', 'mean'),
                count=('pred', 'count')
            ).round(3)

            overall_approval = y_pred.mean()

            # Display Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Overall Approval Rate", f"{overall_approval:.1%}")
            with col3:
                st.metric("Number of Groups", len(group_stats))

            # Bias Detection
            approval_rates = group_stats['approval_rate']
            max_app = approval_rates.max()
            min_app = approval_rates.min()
            disparate_impact = min_app / max_app if max_app > 0 else 0

            st.subheader("Bias Analysis")

            if disparate_impact < 0.8:
                st.error("🔴 HIGH BIAS DETECTED")
                st.markdown(f"**Disparate Impact Ratio**: {disparate_impact:.2f} (Below 0.8 — High Risk)")
            elif disparate_impact < 0.9:
                st.warning("🟡 Moderate Bias Detected")
            else:
                st.success("🟢 Low Bias Detected")

            # Chart
            st.subheader(f"Approval Rate by { 'Gender' if sensitive_attr == 'sex' else sensitive_attr.title() }")
            fig = px.bar(
                group_stats.reset_index(), 
                x='group', 
                y='approval_rate',
                title=f"Approval Rate by {'Gender' if sensitive_attr == 'sex' else sensitive_attr.title()}",
                labels={'approval_rate': 'Approval Rate'},
                color='approval_rate',
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"""
            **Unbiased AI Summary:**
            - The model shows **{'high' if disparate_impact < 0.8 else 'moderate' if disparate_impact < 0.9 else 'low'}** bias with respect to **{'Gender' if sensitive_attr == 'sex' else sensitive_attr.title()}**.
            - This can result in unfair decisions in hiring, loans, and medical care.
            """)

        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.info("👆 Click the button above to start analyzing bias in AI decisions.")

st.caption("Unbiased AI | Google Solution Challenge 2026")