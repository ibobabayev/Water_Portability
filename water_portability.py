import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics

st.set_page_config(
    page_title="Water Portability",
    page_icon=":droplet:",
    layout="wide",
    initial_sidebar_state="expanded",
)


data = pd.read_csv("water_potability.csv")

with st.sidebar:
    option = st.selectbox(
        "",
        ("Homepage", "Preprocessing", "Modeling", "Evaluation and Visualization"), )

if option == 'Homepage':
    st.title("Water Portability data")
    st.image('https://genesiswatertech.com/wp-content/uploads/2019/07/Potable-Water.jpg', width=400)

    st.caption("First 10 observations of data")
    st.table(data.head(10))
    st.caption("Data Types of Columns")
    st.write(data.dtypes)
    st.markdown('Once you are familiar with the data, you can start **preprocessing**')



elif option == 'Preprocessing':
    st.caption("Number of NA values in each column")
    st.write(data.isnull().sum())

    on = st.toggle("Do you want to fill NA values?")

    if on:
        method = st.radio(
            "Choose method", ["Mean", "Median", "Constant value"],
        )

        if method == 'Mean':
            for col in ['ph', 'Sulfate', 'Trihalomethanes']:
                data[col] = data[col].fillna(data[col].mean())

        elif method == 'Median':
            for col in ['ph', 'Sulfate', 'Trihalomethanes']:
                data[col] = data[col].fillna(data[col].mean())

        else:
            number = st.number_input("Insert a number", value=None, placeholder="Type a number...")
            if number:
                for col in ['ph', 'Sulfate', 'Trihalomethanes']:
                    data[col] = data[col].fillna(number)

    st.caption('Summary Statistics')
    st.write(data.describe())

    st.caption('Checking imbalance in Potability(target) column')
    st.write(data['Potability'].value_counts())

    with st.expander("See correlation"):
        correlation_matrix = data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        st.pyplot(plt)

    plt.figure(figsize=(16, 8))
    sns.boxplot(data=data)
    plt.title('Boxplot for Outlier Detection')
    st.pyplot(plt)

    on_scaled = st.toggle("Do you want to scale the model?")
    if on_scaled:
        scale_options = st.multiselect(
            "Choose which method(s) to use for scaling.",
            ["MinMax", "Standard", "Robust"], )

        selected_scalers = []
        for i in scale_options:
            selected_scalers.append(i)

        if selected_scalers:
            if "MinMax" in selected_scalers:
                scaler_minmax = MinMaxScaler()
                scaled_df = pd.DataFrame(scaler_minmax.fit_transform(data), columns=data.columns)
                st.session_state.scaled_df = scaled_df

            if "Standard" in selected_scalers:
                scaler_standard = StandardScaler()
                scaled_df = pd.DataFrame(scaler_standard.fit_transform(data), columns=data.columns)
                st.session_state.scaled_df = scaled_df

            if "Robust" in selected_scalers:
                scaler_robust = RobustScaler()
                scaled_df = pd.DataFrame(scaler_robust.fit_transform(data), columns=data.columns)
                st.session_state.scaled_df = scaled_df

    st.markdown('It looks like your data is ready for **modeling**')



elif option == 'Modeling':
    st.caption('Split the data into training and testing sets')
    if 'scaled_df' in st.session_state:
        scaled_df = st.session_state.scaled_df
        X = scaled_df.drop('Potability', axis=1)
        y = scaled_df['Potability']
        prop = st.slider("Choose the test size for the train-test split", 0, 100, 20, 5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=prop / 100, random_state=None, stratify=y,
                                                            shuffle=True)

    else:
        X = data.drop('Potability', axis=1)
        y = data['Potability']
        prop = st.slider("Choose the test size for the train-test split", 0, 100, 20, 5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=prop / 100, random_state=None, stratify=y,
                                                            shuffle=True)

    method = st.radio(
        "Choose a model to predict the data", ["Logistic Regression", "Random Forest Classifier", "XGBoost"],
    )
    if method == 'LogisticRegression':
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.caption("10 random observations to compare the actual and predicted values")
        st.table(comparison_df.head(10))
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_prob = y_prob

    elif method == 'Random Forest Classifier':
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.caption("10 random observations to compare the actual and predicted values")
        st.table(comparison_df.head(10))
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_prob = y_prob

    else:
        model = XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.caption("10 random observations to compare the actual and predicted values")
        st.table(comparison_df.head(10))
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_prob = y_prob

    st.markdown("Your model is ready. Let's take a look at the **results**")


else:
    if 'y_test' in st.session_state and 'y_pred' in st.session_state:
        report = metrics.classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.write("### Classification Report")
        st.dataframe(report_df)
    if st.session_state.y_prob is not None:
        accuracy_score = metrics.accuracy_score(st.session_state.y_test, st.session_state.y_pred)
        st.write(f'The model made accurate predictions for **{accuracy_score * 100:.2f}%** of all samples.')
        auc_score = metrics.roc_auc_score(st.session_state.y_test, st.session_state.y_prob)
        st.write(
            f'The AUC score of 0.64 indicates that the model has a **{auc_score * 100:.2f}%** chance of distinguishing between positive and negative classes ')
        gini = 2 * metrics.roc_auc_score(st.session_state.y_test, st.session_state.y_pred) - 1
        if gini < 0.3:
            st.write(
                f"A Gini score of **{gini:.3f}** suggests that the model's ability to differentiate between the classes (positive vs. negative) is relatively weak to moderate ")
        elif 0.3 < gini < 0.5:
            st.write(
                f"A Gini score of **{gini:.3f}** suggests that the model has moderate discriminatory power,meaning it can distinguish between the classes (positive vs. negative) with some effectiveness, but there is still room for improvement.")
        else:
            st.write(
                f"A Gini score of **{gini:.3f}** suggests that the model has strong discriminatory power, meaning it is quite effective at differentiating between the classes (positive vs. negative).")

    logit_roc_auc = metrics.roc_auc_score(st.session_state.y_test, st.session_state.y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(st.session_state.y_test, st.session_state.y_prob)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label='AUC = %0.2f' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Potable or Not')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    st.pyplot(plt)

    if st.button("Model is ready!"):
        st.write("Hooray")
