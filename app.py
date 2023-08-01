import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch
import streamlit as st

# Function to preprocess the text
def preprocess_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# Read sentences from CSV or plain text file
def read_sentences(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, header=None)
        sentences = df.iloc[:, 0].tolist()
    else:
        with open(file_path, "r") as file:
            sentences = file.readlines()
    return sentences

# Generate theme names based on topic embeddings
def generate_theme_names(embeddings, num_topics=5):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    kmeans.fit(embeddings)

    theme_names = []
    for i in range(num_topics):
        theme_name = f"Theme {i + 1}"
        theme_names.append(theme_name)

    return theme_names

# Main function
def main():
    st.title("Theme Generator")

    nltk.download("punkt")
    nltk.download("stopwords")

    st.write("Please upload a CSV file containing sentences.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        num_topics = 5  # Number of themes to generate (you can adjust this)

        sentences = read_sentences(uploaded_file)
        preprocessed_sentences = [preprocess_text(sent) for sent in sentences]

        # Use pre-trained BERT model for embeddings
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        with torch.no_grad():
            inputs = tokenizer(preprocessed_sentences, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

        theme_names = generate_theme_names(embeddings, num_topics)

        st.header("Themes:")
        for theme_name in theme_names:
            st.subheader(theme_name)

if __name__ == "__main__":
    main()
