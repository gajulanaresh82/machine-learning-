import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import pickle


from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score,ConfusionMatrixDisplay








model=pickle.load(open(r"C:\Users\Keyush\Desktop\python points\machine-learning-material\ai_elite16\btp.pkl",'rb'))




def make_prediction(feature1,feature2,feature3,feature4,feature5):
    
    
    # Prepare input data
    input_data = np.array([feature1, feature2, feature3, feature4, feature5]).reshape(1, -1)

    # Predict with the loaded model
    prediction = model.predict(input_data)
    
    return prediction

# Main function to run the Streamlit app
def main():
    st.title("Brain Tumor Prediction App")
    st.header("Enter Input Values")

    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)
    feature5 = st.number_input("Feature 5", value=0.0)


    if st.button("Predict"):
        prediction = make_prediction(feature1, feature2, feature3, feature4, feature5)
        st.write("Prediction:", prediction)
        
    

if __name__ == "__main__":
    main()