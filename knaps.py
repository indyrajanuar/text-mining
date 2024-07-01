import streamlit as st
from streamlit_option_menu import option_menu
import joblib
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

        # Text input for new article
        # new_article = st.text_area('Enter the article text here:')

        if st.button('Classify'):
            if upload_file:
                # Transform the new article using the loaded vectorizer
                new_data_tfidf = vectorizer.transform([new_article])
                # Predict the label using the loaded model
                prediction = nb.predict(new_data_tfidf)
                # Display the prediction
                st.write('Predicted Label:', prediction[0])
            else:
                st.write('Please enter an article to classify.')

    elif selected == 'Uji Coba':
        st.markdown('<h1 style="text-align: center;"> Uji Coba </h1>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
