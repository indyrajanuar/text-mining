import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
        st.markdown("""
        ## Klasifikasi Berita Menggunakan Naive Bayes

        **Klasifikasi Berita** adalah proses otomatis untuk mengelompokkan berita ke dalam kategori atau label tertentu berdasarkan konten berita tersebut. Misalnya, sebuah berita bisa diklasifikasikan sebagai "Olahraga", "Politik", "Ekonomi", atau "Teknologi" berdasarkan isinya.

        **Naive Bayes** adalah algoritma pembelajaran mesin yang digunakan untuk tugas klasifikasi. Algoritma ini berdasarkan teorema Bayes dengan asumsi independensi yang sangat sederhana yaitu setiap fitur (kata-kata dalam teks) dianggap independen satu sama lain, walaupun dalam kenyataannya fitur-fitur tersebut bisa saling terkait.

        ### Bagaimana Naive Bayes Bekerja?
        1. **Pengumpulan Data**: Kumpulkan data berita yang sudah dikategorikan dalam berbagai label (misalnya, "Olahraga", "Politik").
        2. **Preprocessing Data**: Lakukan pembersihan data seperti menghapus kata-kata yang tidak penting, mengubah teks menjadi bentuk yang konsisten, dan mengisi nilai yang hilang.
        3. **Ekstraksi Fitur**: Gunakan teknik seperti TF-IDF (Term Frequency-Inverse Document Frequency) untuk mengubah teks berita menjadi fitur numerik yang bisa diproses oleh model.
        4. **Pelatihan Model**: Latih model Naive Bayes menggunakan data pelatihan. Model ini akan mempelajari probabilitas kata-kata muncul di setiap kategori berita.
        5. **Klasifikasi Berita**: Gunakan model yang sudah dilatih untuk mengklasifikasikan berita baru berdasarkan probabilitas yang dipelajari.
        6. **Evaluasi Model**: Ukur kinerja model menggunakan metrik seperti akurasi, presisi, recall, dan f1-score untuk memastikan bahwa model dapat mengklasifikasikan berita dengan baik.

        ### Kelebihan Naive Bayes
        - **Sederhana dan Efisien**: Mudah diimplementasikan dan memerlukan waktu komputasi yang relatif rendah.
        - **Cepat dalam Pelatihan dan Prediksi**: Dapat menangani dataset besar dengan cepat.
        - **Bagus untuk Teks**: Terbukti efektif dalam tugas klasifikasi teks seperti spam filtering dan analisis sentimen.
        
    """)

        st.markdown('<h3 style="text-align: left;"> View Data </h1>', unsafe_allow_html=True)
        
        if upload_file is not None:
            df = pd.read_csv(upload_file)
            st.dataframe(df)

            # Display the number of data points for each label
            label_counts = df['Label'].value_counts()
            st.write('Jumlah data pada setiap label:')
            st.bar_chart(label_counts)
            
            # Display the counts in a table
            st.write("Jumlah data per label:")
            st.write(label_counts)
            
    elif selected == 'Klasifikasi Naive Bayes':
        st.markdown('<h1 style="text-align: center;"> Klasifikasi Naive Bayes </h1>', unsafe_allow_html=True)
        st.write("Berikut merupakan hasil klasifikasi yang di dapat dari pemodelan Naive Bayes")

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

                # Display the metrics
                html_code = f"""
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

                st.markdown(html_code, unsafe_allow_html=True)

    elif selected == 'Uji Coba':
        st.markdown('<h1 style="text-align: center;"> Uji Coba </h1>', unsafe_allow_html=True)
        st.write("Enter an article for classification:")

        # Load the model and vectorizer
        nb, vectorizer = load_model()

        # Input area for new articles
        new_article = st.text_area("New Article")

        if st.button("Classify"):
            if new_article:
                new_article_tfidf = vectorizer.transform([new_article])
                prediction = nb.predict(new_article_tfidf)
                st.write(f"Predicted Label: {prediction[0]}")
            else:
                st.write("Please enter an article to classify.")

if __name__ == "__main__":
    main()
