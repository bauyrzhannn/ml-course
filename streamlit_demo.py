from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="BigData Team: streamlit demo",
    page_icon="🦁",
    layout="wide",
)

st.sidebar.markdown("""
# Мои заметки

1. Введение в библиотеки ML
2. Простые модели
3. Линейные модели и Регуляризация
4. Ансамбли
5. Нейронные сети
6. Трансформеры
7. Обучение без учителя
8. Big Data
9. Проектная деятельность

Github курса:
[github.com/big-data-team/ml-course](https://github.com/big-data-team/ml-course)
""")

st.title("🦁 BigData Team: Streamlit Demo")
st.header("01. Введение в библиотеки ML", divider=True)
st.subheader("Titanic dataset, train sample", divider=True)

train = pd.read_csv("train.csv")
st.write(train)

def preprocess_data(data):
    columns_to_drop = ["Ticket", "PassengerId", "Name", "Cabin"]
    data.drop(columns_to_drop, axis=1, inplace=True)
    
    data["Sex"] = (data["Sex"] == "female").astype(int)
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    data.fillna(-1, inplace=True)

preprocess_data(train)

labels = train["Survived"]
train.drop("Survived", axis=1, inplace=True)

st.subheader("В поисках лучшей kNN модели", divider=True)

col1, col_, col2 = st.columns([0.5, 0.1, 0.4])
with col1:
    n_neighbors = st.slider("Количество соседей", value=5, min_value=1, max_value=25)
    weights = st.selectbox("weights", options=("uniform", "distance"))
    p = st.number_input("distance_p(ower degree)", value=2, min_value=1)
    st.markdown("Больше о параметрах kNN в sklearn: [по ссылке](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)")

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    cross_val_scores = cross_val_score(knn, train, labels, scoring="accuracy", cv=5)

with col2:
    cross_val_score_mean = cross_val_scores.mean()
    delta = None
    
    if "score_history" not in st.session_state:
        st.session_state["score_history"] = []
    
    st.session_state["score_history"].append(cross_val_score_mean)
    
    if "previous_score" in st.session_state:
        delta = round(cross_val_score_mean - st.session_state["previous_score"], 3)
    
    st.write("Результаты")
    st.metric("Accuracy (mean over 5 folds)", round(cross_val_score_mean, 3), delta)
    st.write({"score_mean": cross_val_score_mean, "score_std": cross_val_scores.std()})

    st.session_state["previous_score"] = cross_val_score_mean

st.subheader("История изменений Accuracy", divider=True)
st.line_chart(st.session_state["score_history"])



