import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('regressor.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
norm = data['normalization']

def main():
    st.title("ADMISSION PREDICTION APP")

    st.write("""### We need some information to predict your chances of admission""")


    gre_score = st.text_input("GRE Score","Type Here")
    toefl_score = st.text_input("TOEFL Score","Type Here")
    university_rating = st.slider("University Rating", 1, 5, 1)
    statement_of_purpose_score = st.text_input("Statement of Purpose Score","Type Here")
    letter_of_recommendation_score = st.text_input("Letter of Recommendation Score","Type Here")
    cgpa = st.text_input("CGPA","Type Here")
    research = st.selectbox('Research? 1 = Yes , 0 = No',(1, 0,))
    st.write('You selected:', research)

    ok = st.button("Predict Chance of Admission")

    if ok:
        X = np.array([[gre_score, toefl_score, university_rating, statement_of_purpose_score, 
                        letter_of_recommendation_score, cgpa, research]])
        scaled_X = norm.transform(X)

        score = model.predict(scaled_X)
        percentage = score * 100
        percentage = round(percentage[0])

        st.subheader(f"You have a {percentage}% chance of getting admission")

if __name__=='__main__':
    main()