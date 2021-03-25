import numpy as np
import pandas as pd
import streamlit as st
import pickle
import base64
import warnings
warnings.filterwarnings("ignore")
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')


simple_imputer = load("SimpleImputer.pkl")  # load the preprossing imputer 
model = load("XGBClassifier.pkl")  # load the best classifier 
important_columns = load("important_columns.pkl") # load imp columns

def prediction_F2(X,y):
    y = y.apply(lambda x: 0 if x == 'neg' else 1) 
    col = X.columns
    new = pd.DataFrame(X,columns =col)
    X = X.drop(['br_000','bq_000','bp_000','bo_000','ab_000','cr_000','bn_000','bm_000'],axis=1) # drop the missing columns
    X = simple_imputer.transform(X[important_columns])
    X = pd.DataFrame(X,columns=[important_columns])   
    pred = model.predict(X)                 # predict on Test data
    new['predict'] = pred    
    new['predict'] = new['predict'].apply(lambda x: 'negative' if x == 0 else 'positive')       
    return new

def main():       
    # front end elements of the web page 

    html_temp = """ 
    <div style ="background-color:skyblue;padding:25px"> 
    <h1 style ="color:black;text-align:center;"> Scania Truck Prediction </h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)

    st.title("Upload the file for Prediction") 

    uploaded_file = st.file_uploader("Upload CSV",type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file,na_values='na')
        df = df.replace('na',0)

        def download_link(object_to_download, download_filename, download_link_text):
            if isinstance(object_to_download,pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)
            b64 = base64.b64encode(object_to_download.encode()).decode()
            return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


        if st.checkbox("Start the Prediction"):
            y = df['class']
            x = df.drop('class',axis=1).astype(np.float)
            st.success("Start Predicting...")
            a = prediction_F2(x,y)
            tmp_download_link = download_link(a, 'predicted.csv', 'Click here to download your data!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            st.success("Done...")

    else:
        st.success("Load the file")



if __name__=='__main__': 
    main()
    