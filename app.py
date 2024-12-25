import streamlit as st
from streamlit_option_menu import option_menu
import math, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, classification_report
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

def Home():
    st.title("🍄Klasifikasi Jamur")
    st.header("Model yang digunakan")
    st.write("Single Layer Perceptron")
    st.write("Naive Bayes")
    st.write("Random Forest")
    st.header("Dataset")
    st.write("Dataset yang digunakan adalah dataset jamur yang diperoleh dari situs kaggle.com.")
    st.header("catatan:")
    st.write("Sebelum membuka menu modeling terlebih dahulu buka preprocessing.")

def Understanding():
    st.title("Data Understanding")
    df = pd.read_csv("mushrooms.csv")
    
    st.header("Dataset")
    st.dataframe(df)
    st.write("Jumlah Data : ", len(df.axes[0]))
    st.write("Jumlah Atribut : ", len(df.axes[1]))
    st.write("""
        | **Attribute**                  | **Values (Encoding)**                                                                                               |
        |--------------------------------|----------------------------------------------------------------------------------------------------------------------|
        | **class**                      | edible=e, poisonous=p                                                                                                |
        | **cap-shape**                  | bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s                                                             |
        | **cap-surface**                | fibrous=f, grooves=g, scaly=y, smooth=s                                                                             |
        | **cap-color**                  | brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y                            |
        | **bruises**                    | bruises=t, no=f                                                                                                      |
        | **odor**                       | almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s                                 |
        | **gill-attachment**            | attached=a, descending=d, free=f, notched=n                                                                         |
        | **gill-spacing**               | close=c, crowded=w, distant=d                                                                                       |
        | **gill-size**                  | broad=b, narrow=n                                                                                                   |
        | **gill-color**                 | black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y        |
        | **stalk-shape**                | enlarging=e, tapering=t                                                                                             |
        | **stalk-root**                 | bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?                                               |
        | **stalk-surface-above-ring**   | fibrous=f, scaly=y, silky=k, smooth=s                                                                               |
        | **stalk-surface-below-ring**   | fibrous=f, scaly=y, silky=k, smooth=s                                                                               |
        | **stalk-color-above-ring**     | brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y                                     |
        | **stalk-color-below-ring**     | brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y                                     |
        | **veil-type**                  | partial=p, universal=u                                                                                              |
        | **veil-color**                 | brown=n, orange=o, white=w, yellow=y                                                                                |
        | **ring-number**                | none=n, one=o, two=t                                                                                                |
        | **ring-type**                  | cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z                                |
        | **spore-print-color**          | black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y                               |
        | **population**                 | abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y                                             |
        | **habitat**                    | grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d                                                  |
    """)

    num_columns = len(df.columns)
    cols = 2
    rows = math.ceil(num_columns / cols)

    st.title("Visualisasi Data")
    st.header("Distribusi Antar Fitur dan Target")
    target_column = "class"
    num_features = len(df.columns) - 1
    cols = 2
    rows = -(-num_features // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    feature_columns = [col for col in df.columns if col != target_column]
    for i, column in enumerate(feature_columns):
        sns.countplot(x=df[column], hue=df[target_column], palette='viridis', ax=axes[i])
        axes[i].set_title(f'{column} vs {target_column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frekuensi')

    for i in range(len(feature_columns), len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    st.pyplot(fig)

def Preprocessing():
    st.title("Preprocessing")

    df = pd.read_csv("mushrooms.csv")

    st.header("Label Encoding")
    label_map = {}
    le = LabelEncoder()
    for feature in df:
        df[feature] = le.fit_transform(df[feature].astype(str))
        label_map[feature] = dict(zip(le.classes_, le.transform(le.classes_)))
    st.write(df)
    
    label_table = []
    for feature, labels in label_map.items():
        combined_labels = ", ".join([f"{label} = {encoded}" for label, encoded in labels.items()])
        label_table.append({"Feature": feature, "Labels": combined_labels})

    label_df = pd.DataFrame(label_table)
    st.table(label_df)

    st.header("Feature Selection")
    st.write("Fitur seleksi menggunakan Information Gain dengan threshold = 0.05")
    
    X = df.drop(columns='class')
    y = df['class']
    feature_selected = ['class']
    info_gain = mutual_info_classif(X, y)
    threshold = 0.05
    total = 0
    feature_data = []

    for i, col in enumerate(X.columns):
        feature_data.append({
            "Feature": col,
            "Gain": f"{info_gain[i]:.4f}",
            "Selected": "Yes" if info_gain[i] > threshold else "No"
        })
        if info_gain[i] > threshold:
            feature_selected.append(col)
            total += 1

    feature_df = pd.DataFrame(feature_data)
    st.table(feature_df)
    st.write(f"Total fitur terseleksi: {total}")
    st.write(feature_selected)

    df = df[feature_selected]
    st.session_state["preprocessed_data"] = df

def show_confusion_matrix(cm, title):
    fig_cm, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig_cm)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Akurasi: {accuracy:.4f}")
        st.write(f"Recall: {recall:.4f}")
    with col2:
        st.write(f"Presisi: {precision:.4f}")
        st.write(f"F1-Score: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    show_confusion_matrix(cm, f"Confusion Matrix - {model_name}")

def Modeling():
    st.title("Modeling")
    if 'preprocessed_data' in st.session_state:
        df = st.session_state["preprocessed_data"]
        
        splits = [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]

        for train_ratio, test_ratio in splits:
            train, test = int(train_ratio*100), int(test_ratio*100)
            st.header(f"Data Split: {train}% Training  {test}% Testing")

            y = df['class']
            X = df.drop(columns=['class'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=True, random_state=42)

            st.write(f"Total data testing: {len(X_test)}")

            st.subheader("Single Layer Perceptron")
            param_grid = {'eta0': np.linspace(0.1, 1, 10), 'max_iter': [1000]}
            perceptron = GridSearchCV(Perceptron(), param_grid, scoring='accuracy')
            perceptron.fit(X_train, y_train)
            best_params = perceptron.best_params_
            st.write(f"Learning Rate terbaik: {best_params['eta0']}")
            results = perceptron.cv_results_
            df_results = pd.DataFrame(results)
            accuracy_table = df_results[['param_eta0', 'mean_test_score']]
            st.write("Grid search learning rate:")
            st.dataframe(accuracy_table)
            evaluate_model(perceptron.best_estimator_, X_train, X_test, y_train, y_test, "Perceptron")

            st.subheader("Naive Bayes")
            nb = GaussianNB()
            evaluate_model(nb, X_train, X_test, y_train, y_test, "Naive Bayes")

            st.subheader("Random Forest")
            rf = RandomForestClassifier()
            evaluate_model(rf, X_train, X_test, y_train, y_test , "Random Forest")

def main():
    with st.sidebar :
        page = option_menu ("Wahyudi Firmansyah", ["Home", "Data Understanding", "Preprocessing", "Modeling"], default_index=0)

    if page == "Home":
        Home()
    elif page == "Data Understanding":
        Understanding()
    elif page == "Preprocessing":
        Preprocessing()
    elif page == "Modeling":
        Modeling()

if __name__ == "__main__":
    st.set_page_config(page_title="Single Layer Perceptron")
    main()