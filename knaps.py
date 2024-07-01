import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "Klasifikasi Naive Bayes", "Uji Coba"],
            icons=['house', 'table', 'check2-circle'],
            menu_icon="cast",
            default_index=1,
            orientation='vertical')
        
        upload_file = st.sidebar.file_uploader("Masukkan file csv disini", key=1)
    
    if selected == 'Home':
        st.markdown('<h1 style="text-align: center;"> Home </h1>', unsafe_allow_html=True)
            
    elif selected == 'Klasifikasi Naive Bayes':
        st.markdown('<h1 style="text-align: center;"> Klasifikasi Naive Bayes </h1>', unsafe_allow_html=True)

        # Load the saved model and vectorizer
        nb = joblib.load('naive_bayes_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')

        if upload_file is not None:
            # Read the CSV file
            df = pd.read_csv(upload_file)

            # Check if the CSV file contains the expected column
            if 'Artikel' in df.columns:
                articles = df['Artikel'].fillna('')  # Fill missing values with empty strings
                # Transform the articles using the loaded vectorizer
                articles_tfidf = vectorizer.transform(articles)
                # Predict the labels using the loaded model
                predictions = nb.predict(articles_tfidf)
                # Add predictions to the dataframe
                df['Predicted Label'] = predictions
                # Display the dataframe with predictions
                st.write(df)
            else:
                st.write("The uploaded CSV file does not contain the required 'Artikel' column.")
        else:
            st.write("Please upload a CSV file to classify articles.")

    elif selected == 'Uji Coba':
        st.markdown('<h1 style="text-align: center;"> Uji Coba </h1>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
