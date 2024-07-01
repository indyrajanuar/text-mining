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
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
from gensim.utils import simple_preprocess

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Function for Naive Bayes Classification
def naive_bayes_classification():
    st.markdown('<h1 style="text-align: center;"> Klasifikasi Naive Bayes </h1>', unsafe_allow_html=True)
    
    file_path = 'path/to/your/csvfile.csv'  # update with your file path
    data = pd.read_csv(antaranews.csv)
    
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

# Function for Topic Modelling
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def topic_modelling():
    st.markdown('<h1 style="text-align: center;"> Topic Modelling </h1>', unsafe_allow_html=True)
    
    file_path = 'path/to/your/csvfile.csv'  # update with your file path
    data = pd.read_csv(file_path)
    
    data['processed_artikel'] = data['Artikel'].apply(preprocess_text)
    
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
    X = vectorizer.fit_transform(data['processed_artikel'])
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    
    st.write('Topics:')
    terms = vectorizer.get_feature_names()
    for idx, topic in enumerate(lda.components_):
        st.write(f"Topic {idx}:")
        st.write([terms[i] for i in topic.argsort()[:-11:-1]])

# Main function to create the Streamlit app
def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "Klasifikasi Naive Bayes", "Topic Modelling", "Uji Coba"],
            icons=['house', 'table', 'boxes', 'check2-circle'],
            menu_icon="cast",
            default_index=1,
            orientation='vertical')
        
    if selected == 'Home':
        st.markdown('<h1 style="text-align: center;"> Home </h1>', unsafe_allow_html=True)
            
    elif selected == 'Klasifikasi Naive Bayes':
        naive_bayes_classification()
            
    elif selected == 'Topic Modelling':
        topic_modelling()
            
    elif selected == 'Uji Coba':
        st.markdown('<h1 style="text-align: center;"> Uji Coba </h1>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
