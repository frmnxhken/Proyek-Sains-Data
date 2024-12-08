import streamlit as st
from streamlit_option_menu import option_menu
import math, warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, classification_report
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
warnings.filterwarnings('ignore')

def Home():
    st.title("Klasifikasi Jamur Menggunakan Single Layer Perceptron")

    st.header("Single Layer Perceptron")

    st.header("Dataset")
    st.write("Dataset yang digunakan adalah dataset jamur yang diperoleh dari situs kaggle.com.")

    st.header("Tahapan Proses Klasifikas")
    st.write("1. **Data Understanding**")
    st.write("2. **Preprocessing Data**")
    st.write("3. **Modeling**")
    st.write("4. **Evaluation**")
    st.write("5. **Implementation**")

def Understanding():
    st.title("Data Understanding")
    df = pd.read_csv("mushrooms.csv")
    
    st.header("Dataset")
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
    st.dataframe(df)

    st.write("Jumlah Data : ", len(df.axes[0]))
    st.write("Jumlah Atribut : ", len(df.axes[1]))

    st.header("Check Missing Value")
    df['stalk-root'].replace(['?'], pd.NA, inplace=True)
    st.write(df.isna().sum())
    
    st.header("Describe Data")
    st.write(df.describe())

    num_columns = len(df.columns)
    cols = 2
    rows = math.ceil(num_columns / cols)

    st.title("Visualisasi Data")
    st.header("Distribusi Data Tiap Fitur")

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    df_columns = list(df.columns)
    for i, column in enumerate(df_columns):
        sns.countplot(x=df[column], palette='viridis', ax=axes[i])
        axes[i].set_title(f'Distribusi {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frekuensi')

    for i in range(len(df_columns), len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    st.pyplot(fig)

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
    df['stalk-root'].replace(['?'], np.nan, inplace=True)

    st.header("Label Encoding")
    label_map = {}
    le = LabelEncoder()
    for feature in df:
        nan_mask = df[feature].isna()
        df[feature] = df[feature].fillna('missing')
        df[feature] = le.fit_transform(df[feature].astype(str))
        df[feature][nan_mask] = np.nan
        label_map[feature] = dict(zip(le.classes_, le.transform(le.classes_)))
    st.write(df)
    
    label_table = []
    label_table = []
    for feature, labels in label_map.items():
        combined_labels = ", ".join([f"{label} = {encoded}" for label, encoded in labels.items()])
        label_table.append({"Feature": feature, "Labels": combined_labels})

    label_df = pd.DataFrame(label_table)
    st.table(label_df)

    st.header("Imputation")
    st.write("Imputasi data menggunakan KNNImputer dengan K=5")
    st.write("Sebelum di imputasi")
    st.write(df)

    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    st.write("Sesudah di imputasi")
    st.write(df)

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

    st.header("Data After Preprocessing")
    df = df[feature_selected]
    st.write(df)

    st.session_state["preprocessed_data"] = df

def Modeling():
    st.title("Modeling")
    if 'preprocessed_data' in st.session_state:
        df = st.session_state["preprocessed_data"]
        y = df['class']
        X = df.drop(columns=['class'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

        st.header("Data Splitting")
        st.write(X_train)
        st.write(f"Total data training: {len(X_train)}")

        st.write(X_test)
        st.write(f"Total data testing: {len(X_test)}")

        st.header("Single Layer Perceptron")
        model = Perceptron()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results = pd.concat([X_test.reset_index(drop=True), pd.Series(y_pred)], axis=1)
        st.write(results)
        st.write(f"Jumlah epoch: {model.n_iter_}")

        st.session_state["y_pred"] = y_pred
        st.session_state["y_test"] = y_test

def Evaluation():
    if "y_pred" in st.session_state and "y_test" in st.session_state:
        y_test = st.session_state["y_test"]
        y_pred = st.session_state["y_pred"]
        cm = confusion_matrix(y_test, y_pred)

        st.header("Confusion Matrix")
        fig_cm, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        tn = cm[0, 0]  
        fp = cm[0, 1]
        fn = cm[1, 0] 
        tp = cm[1, 1]
        
        col1, col2 = st.columns(2,vertical_alignment='top')
        with col1:
            st.write(f"True Negatives (TN): {tn}")
        with col2:
            st.write(f"False Positives (FP): {fp}")
        
        col1, col2 = st.columns(2,vertical_alignment='top')
        with col1:
            st.write(f"False Negatives (FN): {fn}")
        with col2:
            st.write(f"True Positives (TP): {tp}")

        st.header("Peformance Metrics")
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100

        col1, col2 = st.columns(2,vertical_alignment='top')
        with col1 :
            st.write(f"Accuracy: {accuracy:.2f}%")
        with col2 :
            st.write(f"Precision: {precision:.2f}%")

        col1, col2 = st.columns(2,vertical_alignment='top')
        with col1 :
            st.write(f"Recall: {recall:.2f}%")
        with col2 :
            st.write(f"F1 Score: {f1:.2f}%")

def Testing():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
        mushroom_features = {
        # "cap_shape": {"bell": "b", "conical": "c", "convex": "x", "flat": "f", "knobbed": "k", "sunken": "s"},
        # "cap_surface": {"fibrous": "f", "grooves": "g", "scaly": "y", "smooth": "s"},
        # "cap_color": {"brown": "n", "buff": "b", "cinnamon": "c", "gray": "g", "green": "r", "pink": "p", "purple": "u", "red": "e", "white": "w", "yellow": "y"},
        "bruises": {"bruises": "t", "no": "f"},
        "odor": {"almond": "a", "anise": "l", "creosote": "c", "fishy": "y", "foul": "f", "musty": "m", "none": "n", "pungent": "p", "spicy": "s"},
        # "gill_attachment": {"attached": "a", "descending": "d", "free": "f", "notched": "n"},
        "gill_spacing": {"close": "c", "crowded": "w", "distant": "d"},
        "gill_size": {"broad": "b", "narrow": "n"},
        "gill_color": {"black": "k", "brown": "n", "buff": "b", "chocolate": "h", "gray": "g", "green": "r", "orange": "o", "pink": "p", "purple": "u", "red": "e", "white": "w", "yellow": "y"},
        # "stalk_shape": {"enlarging": "e", "tapering": "t"},
        "stalk_root": {"bulbous": "b", "club": "c", "cup": "u", "equal": "e", "rhizomorphs": "z", "rooted": "r"},
        "stalk_surface_above_ring": {"fibrous": "f", "scaly": "y", "silky": "k", "smooth": "s"},
        "stalk_surface_below_ring": {"fibrous": "f", "scaly": "y", "silky": "k", "smooth": "s"},
        "stalk_color_above_ring": {"brown": "n", "buff": "b", "cinnamon": "c", "gray": "g", "orange": "o", "pink": "p", "red": "e", "white": "w", "yellow": "y"},
        "stalk_color_below_ring": {"brown": "n", "buff": "b", "cinnamon": "c", "gray": "g", "orange": "o", "pink": "p", "red": "e", "white": "w", "yellow": "y"},
        # "veil_type": {"partial": "p", "universal": "u"},
        # "veil_color": {"brown": "n", "orange": "o", "white": "w", "yellow": "y"},
        # "ring_number": {"none": "n", "one": "o", "two": "t"},
        "ring_type": {"cobwebby": "c", "evanescent": "e", "flaring": "f", "large": "l", "none": "n", "pendant": "p", "sheathing": "s", "zone": "z"},
        "spore_print_color": {"black": "k", "brown": "n", "buff": "b", "chocolate": "h", "green": "r", "orange": "o", "purple": "u", "white": "w", "yellow": "y"},
        "population": {"abundant": "a", "clustered": "c", "numerous": "n", "scattered": "s", "several": "v", "solitary": "y"},
        "habitat": {"grasses": "g", "leaves": "l", "meadows": "m", "paths": "p", "urban": "u", "waste": "w", "woods": "d"}
    }

    def encode_values(values):
        encoder = LabelEncoder()
        encoder.fit(values)
        return encoder

    st.title("Mushroom Feature Selector")
    encoded_results = {}
    user_selections = {}

    for feature, options in mushroom_features.items():
        selected_option = st.selectbox(f"Select {feature}", list(options.keys()))
        value_to_encode = options[selected_option]
        encoder = encode_values(list(options.values()))
        encoded_value = encoder.transform([value_to_encode])[0]
        encoded_results[feature] = encoded_value
    if st.button("Predict"):
        st.write("Fitur yang dipilih")
        for feature, selected in user_selections.items():
            st.write(f"**{feature}**: {selected} ({mushroom_features[feature][selected]})")

        st.write("### Encoded Values")
        st.json(encoded_results)
        input_features = list(encoded_results.values())
        prediction = model.predict([input_features])[0]

        # Display prediction
        st.write("### Prediction")
        st.write("Edible" if prediction == 0 else "Poisonous")
        # st.write(f"**{feature}**: Selected **{selected_option}** ({value_to_encode}), Encoded Value: **{encoded_value}**")
        # st.write("### Encoded Results")
        # st.json(encoded_results)

def main():
    with st.sidebar :
        page = option_menu ("Rundown", ["Home", "Data Understanding", "Preprocessing", "Modeling", "Evaluation", "Testing"], default_index=0)

    if page == "Home":
        Home()
    elif page == "Data Understanding":
        Understanding()
    elif page == "Preprocessing":
        Preprocessing()
    elif page == "Modeling":
        Modeling()
    elif page == "Evaluation":
        Evaluation()
    elif page == "Testing":
        Testing()

if __name__ == "__main__":
    st.set_page_config(page_title="Single Layer Perceptron")
    main()