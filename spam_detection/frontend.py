import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

#preporcees function 
#practical view

def transform_function(x):
  #converting all data to lower case
  x=x.lower()
  # converting/tokenize string into words
  x=nltk.word_tokenize(x)
  #creating loop for check isalnum(alphabetic numeric alue)
  y=[]
  for i in x:
    if i.isalnum():
      y.append(i)
  x=y[:]
  #clearing y list so we can append again
  y.clear()
  #creating loop for checking stopwords and punctuation
  for i in x:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  x=y[:]
  y.clear()
  # cretaed loop for checking stremming data
  for i in x:
    y.append(ps.stem(i))
  return " ".join(y)

#-----------------------------------------------------------------------------
#rb means read binary mode
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("EMAIL/SMS SPAM DETECTOR")

input_sms=st.text_area("Enter the message ")

if st.button('Predict'):

    #preprocess code 

    transformed_sms=transform_function(input_sms)
    #vectorize
    vector_input= tfidf.transform([transformed_sms])

    #predict
    result=model.predict(vector_input)[0]

    #display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("Not Spam")
