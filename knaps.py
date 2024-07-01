import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

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
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
            
    elif selected == 'Klasifikasi Naive Bayes':
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
            
    elif selected == 'Topic Modelling':
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
            
    elif selected == 'Uji Coba':
        st.markdown('<h1 style="text-align: center;"> Website Klasifikasi Hipertensi </h1>', unsafe_allow_html=True)
 
if __name__ == "__main__":
    main()
