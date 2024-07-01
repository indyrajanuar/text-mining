import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def split_data(data):
    # Split data into features and target
    X = data['Artikel']
    y = data['Label']
    # Handle missing values before vectorization
    X = X.fillna('')  # Replace missing values with empty strings
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_model():
    # Load pre-trained model and vectorizer
    model = joblib.load('naive_bayes_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "Klasifikasi Naive Bayes", "Uji Coba"],
            icons=['house', 'table', 'check2-circle'],
            menu_icon="cast",
            default_index=0,
            orientation='vertical')
        
        upload_file = st.sidebar.file_uploader("Masukkan file csv disini", key=1)
    
    if selected == 'Home':
        st.markdown('<h1 style="text-align: center;"> Home </h1>', unsafe_allow_html=True)
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.dataframe(df)
            
    elif selected == 'Klasifikasi Naive Bayes':
        st.markdown('<h1 style="text-align: center;"> Klasifikasi Naive Bayes </h1>', unsafe_allow_html=True)

        if upload_file is not None:
            df = pd.read_csv(upload_file)

            # Load the model and vectorizer
            nb, vectorizer = load_model()

            # Check if the CSV file contains the expected columns
            if 'Artikel' in df.columns and 'Label' in df.columns:
                # Split data
                X_train, X_test, y_train, y_test = split_data(df)
                
                # Transform the training and testing data using the loaded vectorizer
                X_train_tfidf = vectorizer.transform(X_train)
                X_test_tfidf = vectorizer.transform(X_test)

                # Train a new MultinomialNB model
                nb = MultinomialNB()
                nb.fit(X_train_tfidf, y_train)
                
                # Predict the labels for the test set
                y_pred = nb.predict(X_test_tfidf)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                # Generate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
                plt.title('Confusion Matrix')
                st.pyplot(plt.gcf())  # Pass the current figure to st.pyplot()
                # Clear the current plot to avoid displaying it multiple times
                plt.clf()

                # Generate classification report
                with np.errstate(divide='ignore', invalid='ignore'):  # Suppress division by zero warning
                    report = classification_report(y_test, y_pred, zero_division=0)

                # html_code = f"""
                <br>
                <table style="margin: auto;">
                    <tr>
                        <td style="text-align: center; border: none;"><h5>Accuracy</h5></td>
                        <td style="text-align: center; border: none;"><h5>Precision</h5></td>
                        <td style="text-align: center; border: none;"><h5>Recall</h5></td>
                        <td style="text-align: center; border: none;"><h5>F1- Score</h5></td>
                    </tr>
                    <tr>
                        <td style="text-align: center; border: none;">{accuracy * 100:.2f}%</td>
                        <td style="text-align: center; border: none;">{precision * 100:.2f}%</td>
                        <td style="text-align: center; border: none;">{recall * 100:.2f}%</td>
                        <td style="text-align: center; border: none;">{f1 * 100:.2f}%</td>
                    </tr>
                </table>
                """

                # Optionally, classify new articles from the uploaded file
                if 'Artikel' in df.columns:
                    articles = df['Artikel'].fillna('')  # Fill missing values with empty strings
                    articles_tfidf = vectorizer.transform(articles)
                    predictions = nb.predict(articles_tfidf)
                    df['Predicted Label'] = predictions
                    st.write(df)
            else:
                st.write("The uploaded CSV file does not contain the required 'Artikel' or 'Label' columns.")
        else:
            st.write("Please upload a CSV file to classify articles.")

    elif selected == 'Uji Coba':
        st.markdown('<h1 style="text-align: center;"> Uji Coba </h1>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
