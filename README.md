# Admission-Prediction
Create a Machine Learning model that predict the chances of a student getting admission

![Admission-process](https://user-images.githubusercontent.com/58162911/131233359-cb160ce2-5d77-4c7b-ba3b-8d556dc4491b.jpg)
Image Credit: Donetsk National Medical University

## Business Understanding
The main goal of this project is to collect and analyze student data that contain featutres that determines the likelihood of getting admission for Masters in the university. We will help students check the possibility of getting admitted with their different exam scores.

## Analytical Approach
This is a supervised regression problem where we will determine the chances of getting admission. 0 means no chance, 1 shows a very high chance and 0.5 is the cutoff. We will use the different regression algorithms to solve this problem.

## Data Collection
The data was gotten from Kaggle https://www.kaggle.com/mohansacharya/graduate-admissions.

## Data Understanding
The data contains the following features:
* TOFEL and GRE scores
* University Ratings
* CGPA of applicant
* Scores out of 5 for the Statement of Purpose (SOP) and Letter of Recommendation (LOR)
* Whether the graduate course is Researched based or not.

## Modeling
* I scaled all the values of the features to keep them be in the same range for easy model building
* **Linear Regression**, **Lasso Regression**, **Linear SVR**, **Thiel San Regressor** , **Least Angle Regression**,  **Ridge Regression**, **Elastic Net**, **Bayesian Regressor** and **Random Forest Regression** models were all built.
* **Mean Absolute Error (MAE)** and **Root mean squared error (RMSE)** were the metrics used to evaluate the performance of the model.
* Linear Regression has the lowest MAE and RMSE, so it is the chosen model.

## [Web Application](https://share.streamlit.io/victorumunna/admission-prediction/main/app.py)
I built a web application using Streamlit and deployed it to Streamlit Cloud.
The web app : [Admsission Prediction App](https://share.streamlit.io/victorumunna/admission-prediction/main/app.py)
