import streamlit as st
import pickle
import numpy as np


def load_model():
    with open("regressor.pkl", "rb") as file:
        data = pickle.load(file)
    return data


data = load_model()

model = data["model"]
norm = data["normalization"]


def main():
    st.title("ADMISSION PREDICTION APP")

    st.write("""### We need some information to predict your chances of admission""")

    gre_score = st.number_input("GRE Score", min_value=0, max_value=340, value=0)
    toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=0)
    university_rating = st.slider("University Rating", 1, 5, 1)
    statement_of_purpose_score = st.number_input(
        "Statement of Purpose Score", min_value=0.0, max_value=5.0, value=0.0
    )
    letter_of_recommendation_score = st.number_input(
        "Letter of Recommendation Score", min_value=0.0, max_value=5.0, value=0.0
    )
    cgpa = st.number_input("CGPA", min_value=0.00, max_value=10.00, value=0.00)
    research = st.selectbox(
        "Research?  1 = Yes , 0 = No",
        (
            1,
            0,
        ),
    )
    if research == 1:
        st.write("You selected: Yes")
    else:
        st.write("You selected: No")

    ok = st.button("Predict Chance of Admission")

    if ok:
        X = np.array(
            [
                [
                    gre_score,
                    toefl_score,
                    university_rating,
                    statement_of_purpose_score,
                    letter_of_recommendation_score,
                    cgpa,
                    research,
                ]
            ]
        )
        scaled_X = norm.transform(X)

        score = model.predict(scaled_X)
        percentage = score * 100
        percentage = round(percentage[0])

        st.subheader(f"You have {percentage}% chance of getting admission")


if __name__ == "__main__":
    main()
