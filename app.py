import streamlit as st
from ml_prediction.predictions import predict_lstm, predict_gpt2
import pandas as pd

def main():
    st.title("Next Word Prediction for Bangla")
    select_box = st.selectbox(
        "Select Model",
        ('LSTM', 'GPT2')
    )
    ti_value = 'সে একদিন'
    title = st.text_input('Insert Some Bangla Text', ti_value)
    if st.button("Predict"):
        if select_box == 'LSTM':
            res = predict_lstm(title)
            # res = pd.DataFrame({'Word': ['a', 'b', 'c', 'd', 'e'], 'Prob': [1,2,3,4,5]})
            st.subheader("Result for LSTM:")
            st.table(res)
        elif select_box == 'GPT2':
            res = predict_gpt2(title)
            st.subheader("Result of GPT2:")    
            st.table(res)
        else:
            st.write("Select correct model")


if __name__ == '__main__':
    main()