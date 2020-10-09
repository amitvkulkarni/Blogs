import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
st.title('Streamlit example barplot')

@st.cache
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
    df = df.groupby(['country','year'])['lifeExp'].mean().sort_values(ascending=False).reset_index()
    return df

#data_load_state = st.text('Loading data...')
data = load_data()

#data_load_state.text("Done! (using st.cache)")

#if st.checkbox('Show raw data'):
#    st.subheader('Raw data')
#    st.write(data)


#filtered_df = filtered_df.groupby(['country'])['lifeExp'].mean().sort_values(ascending=False).reset_index()
#filtered_df_plot = filtered_df.head(20)
    
year_to_filter = st.slider('year', 1972, data['year'].min(), data['year'].max(),5)
filtered_data = data[data["year"] == year_to_filter]
filtered_data_plot = filtered_data.head(20) 
filtered_data_year = filtered_data[['country','lifeExp']]
st.subheader('Barplot: Country vs LifeExp')


plt.figure(figsize=(30,10))
sns.barplot(x="country", y="lifeExp", data = filtered_data_plot)
st.pyplot()



