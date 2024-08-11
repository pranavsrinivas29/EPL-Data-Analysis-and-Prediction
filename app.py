import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
import numpy as np

# Load custom CSS from the assets folder
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# Load the saved ARIMA models
# Load the saved models (for future use)
with open('model pickel/best_model_home_goals.pkl', 'rb') as f:
    loaded_model_home = pickle.load(f)

with open('model pickel/best_model_away_goals.pkl', 'rb') as f:
    loaded_model_away = pickle.load(f)
    
    
# Load the saved probabilities for champions and runners-up
with open('model pickel/champion_probabilities.pkl', 'rb') as f:
    loaded_champion_prob = pickle.load(f)

with open('model pickel/runner_up_probabilities.pkl', 'rb') as f:
    loaded_runner_up_prob = pickle.load(f)
    

df_yearly = pd.read_csv('premier-league-matches.csv', index_col='Season_End_Year')
df_yearly.index = df_yearly.index.map(str)

epl_season_data = pd.read_csv('epl_season_1993_2024.csv', index_col='Season_End_Year')
epl_season_data.index = epl_season_data.index.map(str)

epl_teams = pd.read_csv('premier-league-matches.csv')
epl_teams.index = epl_teams.index.map(str)

# Display the Premier League logo
st.image('assets/logo.png', width=180)


st.markdown(
    f"""
    <div style="background-color:#1e1e1e;padding:20px;border-radius:10px;box-shadow:0px 4px 8px rgba(0, 0, 0, 0.2);">
        <h2 style="color:#FFD700;text-align:center;margin-bottom:20px;font-family:Arial, sans-serif;">ENGLISH PREMIER LEAGUE</h2>
        <h4 style="color:#FFFFFF;text-align:center;font-family:Georgia, serif;line-height:1.6;">
            The Premier League is the highest level of the English football league system. 
            Contested by 20 clubs, it operates on a system of promotion and relegation with the English Football League (EFL).
        </h4>
    </div>
    """, unsafe_allow_html=True
)



# Standardize team names
epl_teams['Home'] = epl_teams['Home'].replace({
    "Manchester Utd": "Manchester United",
    "Nott'ham Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    # Add other replacements if necessary
})
epl_teams['Away'] = epl_teams['Away'].replace({
    "Manchester Utd": "Manchester United",
    "Nott'ham Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    # Add other replacements if necessary
})

# Ensure the index is named 'Season_End_Year'
epl_season_data.index.name = 'Season_End_Year'

# Convert the index to an integer if it's not already
epl_season_data.index = epl_season_data.index.astype(int)



forecast_plot = 'Prophet_forecast_plot.png'

# Paths to saved plots
HAwins_ratio = 'H_A_Wins_ratios_plot.png'  # Replace with your actual file path
AvgG_per_M = 'Avg_g_per_m.png'  # Replace with your actual file path


# Simulate future seasons
def predict_future_outcome(loaded_champion_prob, loaded_runner_up_prob, future_periods=5):
    np.random.seed(42)
    predicted_champions = []
    predicted_runners_up = []

    # Align teams and probabilities
    all_teams = set(loaded_champion_prob.index).union(set(loaded_runner_up_prob.index))
    teams = sorted(list(all_teams))

    # Ensure probabilities are aligned with the teams
    champion_probs = np.array([loaded_champion_prob.get(team, 0) for team in teams])
    runner_up_probs = np.array([loaded_runner_up_prob.get(team, 0) for team in teams])

    # Normalize the probabilities to sum to 1
    champion_probs = champion_probs / champion_probs.sum()
    runner_up_probs = runner_up_probs / runner_up_probs.sum()

    for _ in range(future_periods):
        champion = np.random.choice(teams, p=champion_probs)
        runner_up = np.random.choice(teams, p=runner_up_probs)

        # Avoid the same team being both champion and runner-up
        while runner_up == champion:
            runner_up = np.random.choice(teams, p=runner_up_probs)

        predicted_champions.append(champion)
        predicted_runners_up.append(runner_up)

    return predicted_champions, predicted_runners_up


# Function to plot a pie chart for Home and Away Goals
def plot_goals_pie_chart(df, year):
    fig, ax = plt.subplots()
    df_year = df.loc[year]
    labels = ['Home Goals', 'Away Goals']
    sizes = [df_year['Total_HomeGoals'], df_year['Total_AwayGoals']]
    colors = ['blue', 'orange']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Home vs Away Goals in {year}')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig

# Function to plot a bar graph for Home Wins and Away Wins
def plot_wins_bar_chart(df, year):
    fig, ax = plt.subplots()
    df_year = df.loc[year]
    labels = ['Home Wins', 'Away Wins']
    wins = [df_year['Home_Wins'], df_year['Away_Wins']]
    ax.bar(labels, wins, color=['green', 'red'])
    ax.set_ylabel('Wins')
    ax.set_title(f'Home Wins vs Away Wins in {year}')
    return fig

# Calculate the probabilities

# Calculate historical win and runner-up rates
champion_counts = epl_season_data['Champion'].value_counts()
runner_up_counts = epl_season_data['Runners'].value_counts()

total_seasons = len(epl_season_data)
champion_prob = champion_counts / total_seasons
runner_up_prob = runner_up_counts / total_seasons


# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Forecasting", "Data Analysis"], index=0)  # Default to "Forecasting" (index=0)

if page == "Forecasting":
    # Forecasting Page
    st.title('Yearly Goals Forecasting')

    # Display the plot
    st.write('### Forecast Plot:')
    
    # Select future periods for forecast
    future_periods = st.number_input('Enter number of future periods to forecast:', min_value=1, max_value=10, value=1)
    
    predicted_champions, predicted_runners_up = predict_future_outcome(loaded_champion_prob, loaded_runner_up_prob, future_periods)


    # Generate future years based on input
    last_year = int(epl_season_data.index.max())
    future_years = np.arange(last_year + 1, last_year + 1 + future_periods)
    
    # Forecast using the loaded models
    forecast_home_goals = np.ceil(loaded_model_home.predict(start=len(epl_season_data), end=len(epl_season_data) + future_periods - 1))
    forecast_away_goals = np.ceil(loaded_model_away.predict(start=len(epl_season_data), end=len(epl_season_data) + future_periods - 1))
    forecast_total_goals = forecast_home_goals + forecast_away_goals  # Ensure total goals is the sum of home and away goals

    # Compile the forecasted data into a DataFrame
    predictions = pd.DataFrame({
        "Season_End_Year": future_years.astype(str),
        "Predicted_Total_Goals": forecast_total_goals.astype(int),
        "Predicted_HomeGoals": forecast_home_goals.astype(int),
        "Predicted_AwayGoals": forecast_away_goals.astype(int)
    })
    
    # Predict future champions and runners-up
    predicted_champions, predicted_runners_up = predict_future_outcome(loaded_champion_prob, loaded_runner_up_prob, future_periods)

    # Add champion and runner-up predictions to the DataFrame
    predictions['Predicted_Champion'] = predicted_champions
    predictions['Predicted_Runner_Up'] = predicted_runners_up


    # Convert the DataFrame to HTML and hide the index
    predictions_html = predictions.to_html(index=False)

    # Display the HTML table without the index
    st.markdown(predictions_html, unsafe_allow_html=True)
            
elif page == "Data Analysis":
    # Data Analysis Page
     # Data Analysis Page
    st.title('Yearly Goals Data Analysis')

    # Display the DataFrame
    st.write('### Yearly Data:')
    st.dataframe(epl_season_data)
    
    # Dropdown to select a year
    year = st.selectbox('Select a Year', epl_season_data.index)

    # Display plots based on selected year
    col1, col2 = st.columns(2)


    with col1:
        st.write('### Goals Distribution:')
        st.pyplot(plot_goals_pie_chart(epl_season_data, year))

    with col2:
        st.write('### Wins Distribution:')
        st.pyplot(plot_wins_bar_chart(epl_season_data, year))
    
    #epl_teams
    # Extract unique team names from the 'Home' and 'Away' columns
    home_teams = epl_teams['Home'].unique()
    away_teams = epl_teams['Away'].unique()
    
    # Combine and sort the list of teams
    teams = sorted(set(home_teams).union(set(away_teams)))
    
    team = st.selectbox('Select a Team', teams)
    
    # Calculate the number of times the selected team has been a winner or runner-up
    winners_count = epl_season_data[epl_season_data['Champion'] == team].shape[0]
    runners_up_count = epl_season_data[epl_season_data['Runners'] == team].shape[0]
    
    if winners_count > 0 or runners_up_count > 0:
        st.markdown(f"""
        <div style="background-color:#e8f4f8;padding:15px;border-radius:10px;">
            <h2 style="color:#1f77b4;text-align:center;">{team} Achievements</h2>
            <ul style="list-style-type:none;padding-left:0;">
                <li style="font-size:20px;color:#2ca02c;"><strong>🏆 Number of Times Champions:</strong> {winners_count}</li>
                <li style="font-size:20px;color:#ff7f0e;"><strong>🥈 Number of Times Runners-Up:</strong> {runners_up_count}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#f8d7da;padding:15px;border-radius:10px;">
            <h2 style="color:#d9534f;text-align:center;">{team} Achievements</h2>
            <p style="font-size:20px;color:#721c24;text-align:center;"><strong>{team} has never been a Champion or Runners-Up.</strong></p>
        </div>
        """, unsafe_allow_html=True)
            
   
    # Display saved Plot 1 in the first column
    #with col1:
    st.write('### H/A wins ratio ')
    st.image(HAwins_ratio, width=800)
        #st.image(plot1_path, )

    # Display saved Plot 2 in the second column
    #with col2:
    st.write('### Avg G/M')
    st.image(AvgG_per_M, width=800)

    
    
    