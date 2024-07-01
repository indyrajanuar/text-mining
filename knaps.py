import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Function for Naive Bayes Classification
def naive_bayes_classification():
    st.markdown('<h1 style="text-align: center;"> Klasifikasi Naive Bayes </h1>', unsafe_allow_html=True)
    
    file_path = 'antaranews.csv'  # update with your file path
    data = pd.read_csv(file_path)
    
    X = data['Artikel']
    y = data['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.fillna('')
    X_test = X_test.fillna('')
    
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    
    y_pred = nb.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy * 100:.2f}%')
    st.write('Classification Report:')
    st.text(classification_report(y_test, y_pred))
    st.write('Confusion Matrix:')
    st.text(confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(nb, 'naive_bayes_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Main function to create the Streamlit app
def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "Klasifikasi Naive Bayes", "Uji Coba"],
            icons=['house', 'table', 'check2-circle'],
            menu_icon="cast",
            default_index=1,
            orientation='vertical')
        
    if selected == 'Home':
        st.markdown('<h1 style="text-align: center;"> Home </h1>', unsafe_allow_html=True)
            
    elif selected == 'Klasifikasi Naive Bayes':
        naive_bayes_classification()
            
    elif selected == 'Uji Coba':
        st.markdown('<h1 style="text-align: center;"> Uji Coba </h1>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
