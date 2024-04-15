import os
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import PercentFormatter

load_dotenv(find_dotenv())
client = OpenAI()

def truncate_text(text, max_length):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def get_user_input(prompt, default_value=""):
    return st.text_area(prompt, default_value)

# Use the get_user_input function to get the prompt
text = get_user_input("Enter your question", "")

def generate_recommendation(business_data):
    
    # Generate prompt for ChatGPT based on reviews and sentiment scores
    prompt = ""
    for index, row in business_data.iterrows():
        review = row.iloc[0]
        sentiment_score = row.iloc[1]
        # Truncate the review if it exceeds a certain length
        review = truncate_text(review, 1200)
        prompt += f"Independant Variables: {review}\nWeight: {sentiment_score}\n"

    # Truncate prompt to a maximum length of 4096 tokens
    prompt = truncate_text(prompt, 4096)

    # Generate completion using OpenAI API
    # client.api_key = 
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": text,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=1000,
    )

    if response.choices[0].finish_reason == "stop":
        recommendation = response.choices[0].message.content
    else:
        print("Response details:", response)
        recommendation = "Error generating recommendation"

    return recommendation

def main():
    st.title("Business Reviews")

    # Upload CSV file
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        if file is not None:
            if not file.name.endswith(".csv"):
                st.error("Error: Please upload a CSV file.")
                return
            try:
                business_data = pd.read_csv(file)

                # Display uploaded data
                st.subheader("Uploaded Reviews")
                st.write(business_data)

                # Check if required columns exist
                if len(business_data.columns) >= 2:
                    st.subheader("Reviews Summary")
                    recommendation = generate_recommendation(business_data)

                    # Display recommendation
                    st.write("Summary Recommendation:")
                    st.write(str(recommendation))  # Convert recommendation to string
                else:
                    st.write(
                        "Error: The uploaded CSV file does not contain all the required columns."
                    )
            except pd.errors.EmptyDataError:
                st.error("Error: The uploaded CSV file is empty.")
            except Exception as e:
                st.error(f"An error occured: {e}")

if __name__ == "__main__":
    main()