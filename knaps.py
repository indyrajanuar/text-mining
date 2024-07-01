import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def main():
    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            ["Home", "Klasifikasi Naive Bayes", "Topic Modelling", "Uji Coba"],
            icons=['house', 'table', 'boxes', 'boxes', 'check2-circle'],
            menu_icon="cast",
            default_index=1,
            orientation='vertical')
        
        
    if selected == 'Home':
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
            
    elif selected == 'Klasifikasi Naive Bayes':
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
            
    elif selected == 'Topic Modelling':
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
            
    elif selected == 'Uji Coba':
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
 
if __name__ == "__main__":
    main()
