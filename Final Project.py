#!/usr/bin/env python
# coding: utf-8

# ### FINAL PROJECT
# ## Keith Howard
# ### 12/10/2022



import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


st.set_page_config(layout="wide")
st.title('Are you a LinkedIn User? Lets find out!')


col1, col2 = st.columns([1, 1])



s = pd.read_csv('social_media_usage.csv')
s = pd.DataFrame(s)

def clean_sm(x):
   global x_clean
   x_clean = pd.DataFrame(data = np.where(x == 1, 1, 0))

sm_li = s["web1h"] 

clean_sm(sm_li)

s.insert(2, "sm_li", x_clean, True)  

ss = s[['sm_li', 'income', 'educ2', 'par', 'marital','gender','age']].copy() 

drop_items = ss[ (ss['income'] > 9) | (ss['educ2'] > 8) | (ss['age'] > 98) ].index
ss.drop(drop_items , inplace=True)

y = np.array(ss["sm_li"]) 

x = ss[ss.columns[1:8]].to_numpy() 

x_train, x_test, y_train, y_test =    train_test_split(x, y, test_size=0.2, random_state=0)

#Creating the model
model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr',
                           random_state=0, class_weight='balanced')
model.fit(x_train, y_train)
## STREAMLIT STUFF ###


#st.sidebar.markdown("## Description of Values")
##st.sidebar.markdown("""Here are a description of the values 
#test
#test
#test""")

#income = st.number_input('What is your income level (1-9)?',  min_value = 1, max_value = 9)


income_text = ("""
	1.	Less than $10,000
	2.	10 to under $20,000
	3.	20 to under $30,000
	4.	30 to under $40,000
	5.	40 to under $50,000
	6.	50 to under $75,000
	7.	75 to under $100,000
	8.	100 to under $150,000, OR
	9.	$150,000 or more?

""")

educ2_text =("""
	1.	Less than high school (Grades 1-8 or no formal schooling)
	2.	High school incomplete (Grades 9-11 or Grade 12 with NO diploma)
	3.	High school graduate (Grade 12 with diploma or GED certificate)
	4.	Some college, no degree (includes some community college)
	5.	Two-year associate degree from a college or university
	6.	Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)
	7.	Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)
	8.	Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)

""")

par_text = ("""
	1.	Yes
	2.	No

""")

marital_text = ("""
	1.	Married
	2.	Living with a partner
	3.	Divorced
	4.	Separated
	5.	Widowed
	6.	Never been married
    
""")

gender_text = ("""
	1.	Male
	2.	Female
	3.	Other

""")
with col1:
	income = st.radio(
		"What's your income?",
		(1,2,3,4,5,6,7,8,9),help = income_text, horizontal = True)
	expander_income = st.expander(label='Description of Values (Expand Me)')
	with expander_income:
		st.write(income_text)

	educ2 = st.radio(
		"What is the highest level of education received?",
		(1,2,3,4,5,6,7,8),help = educ2_text, horizontal = True)
	expander_educ2 = st.expander(label='Description of Values (Expand Me)')
	with expander_educ2:
		st.write(educ2_text)
	#educ2 = st.number_input("What is the highest level of education received? (1-8)",  min_value = 1, max_value = 8)

	par = st.radio(
		"Are you a parent of a child under 18 living in your home?",
		(1,2),help = par_text, horizontal = True)
	expander_par = st.expander(label='Description of Values (Expand Me)')
	with expander_par:
		st.write(par_text)
	#par = st.number_input("Are you a parent of a child under 18 living in your home? (1-2)", min_value = 1, max_value = 2)

	marital = st.radio(
		"What is your current marital status?",
		(1,2,3,4,5,6),help = marital_text, horizontal = True)
	expander_marital = st.expander(label='Description of Values (Expand Me)')
	with expander_marital:
		st.write(marital_text)

	#marital = (st.number_input("What is your current marital status? (1-6)",  min_value = 1, max_value = 6))

	gender = st.radio(
		"What is your gender",
		(1,2,3),help = gender_text, horizontal = True)
	#gender = st.number_input("What is your gender? (1-3)",  min_value = 1, max_value = 3)
	expander_gender = st.expander(label='Description of Values (Expand Me)')
	with expander_gender:
		st.write(gender_text)

	age = (st.number_input("What is your age?",  min_value = 18, max_value = 97))



pred_array = [income,educ2,par,marital,gender,age]
pred_array = [ int(x) for x in pred_array ]
pred2 = np.array([pred_array])
user_pred = 100*np.round(model.predict_proba(pred2), 4)
user_pred2 = user_pred[0].tolist()
user_pred3 = round(user_pred2[1],0)
#st.write(f"The liklihood of you using LinkedIn is {user_pred2}%")

with col2:
#	st.write(user_pred[0,1])
	if user_pred[0,1] > 50: 
		col2.subheader ("You're most likely a LinkedIn User!")
	else:
	   col2.subheader ("You're likely not a LinedIn User. ")
	mycolors = ["white", "#2878e0"]
	fig,results = plt.subplots()
	results.pie(user_pred2, startangle = 90, colors = mycolors)
	results.text(0,0, f"{user_pred3}%", bbox=dict(facecolor='white', alpha=1), fontsize = 35, horizontalalignment='center', verticalalignment='center')
	st.pyplot(fig)


st.text("Author: Keith Howard | Georgetown MSBA, Programming II")