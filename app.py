import transformers
import streamlit as st
from ml_prediction.predictions import predict_lstm, predict_gpt2

def main():
    st.title("Next Word Prediction for Bangla")
    select_box = st.selectbox(
        "Slect Model",
        ('LSTM', 'GPT2')
    )

    title = st.text_input('Insert Some Bangla Text', 'সে একদিন')
    if st.button("Predict"):
        if select_box == 'LSTM':
            res = predict_lstm(title)
            st.subheader("Result for LSTM:")
            st.table(res)
        elif select_box == 'GPT2':
            res = predict_gpt2(title)
            st.subheader("Result for GPT2:")
            st.table(res)
        else:
            st.write("Select correct model")


if __name__ == '__main__':
    main()