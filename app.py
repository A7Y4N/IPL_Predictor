import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
import sklearn
warnings.filterwarnings("ignore")

st.set_page_config(page_title="IPL 2025 Match Predictor")  
st.title("IPL 2025 Match Predictor")
st.write("Welcome to the match prediction app!")  
col1, col2 = st.columns(2) 

teams = ['Chennai Super Kings',
 'Delhi Capitals',
 'Gujarat Titans',
 'Kolkata Knight Riders',
 'Lucknow Super Giants',
 'Mumbai Indians',
 'Punjab Kings',
 'Rajasthan Royals',
 'Royal Challengers Bangalore',
 'Sunrisers Hyderabad']

city = ['Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur',
       'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban',
       'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Rajkot', 'Kanpur', 'Bengaluru', 'Indore', 'Dubai', 'Sharjah',
       'Navi Mumbai', 'Lucknow', 'Guwahati', 'Mohali']

pipe = pickle.load(open('pipe.pkl', 'rb'))
with col1:
    batting_team = st.selectbox("Select the Batting Team", sorted(teams))
with col2:
    bowling_team = st.selectbox("Select the Bowling Team", sorted(teams))

selected_city = st.selectbox("Select the City", sorted(city))
target = st.number_input("Target Score", min_value=0, max_value=500, value=0)

col3,col4,col5 = st.columns(3)
with col3:
    current_score = st.number_input("Current Score", min_value=0, max_value=500, value=0)
with col4:
    overs = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
with col5:
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=0)


runs_left = target - current_score
balls_left = 120 - (overs*6)
wickets = 10 - wickets
crr = current_score/overs
rrr = (runs_left*6)/balls_left


if st.button("Predict Probability"):
    if batting_team == bowling_team:
        st.warning("Batting and Bowling teams cannot be the same!")
    else:
        input_df = pd.DataFrame({ 'batting_team':[batting_team], 'bowling_team':[bowling_team],
        'city': [selected_city], 'runs_left':[runs_left], 'balls_left':[balls_left], 'wickets' :[wickets],
        'total_runs_x' :[target],'crr':[crr],'rrr':[rrr]})
        result = pipe. predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.text(batting_team + "- " + str(round(win*100)) + "%")
        st.text(bowling_team + "- " + str(round(loss*100)) + "%")
st.markdown("""
## About
This app predicts the probability of a team winning an IPL match based on various factors such as the teams playing, the city, the target score, current score, overs completed, and wickets lost.
""")



