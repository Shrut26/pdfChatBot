import streamlit as st
from PyPDF2 import PdfReader
import os
import groq
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = Groq(api_key='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')

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_tfidf_index(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def retrieve_passages(query, texts, vectorizer, tfidf_matrix, k=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    top_indices = similarities.argsort()[0][-k:][::-1]
    relevant_passages = []
    for i in top_indices:
        if i < len(texts):
            relevant_passages.append(texts[i])
    
    return relevant_passages


st.title("PDF Chatbot")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    vectorizer, tfidf_matrix = create_tfidf_index([pdf_text])
    user_queries = []
    if "user_queries" not in st.session_state:
        st.session_state.user_queries = 0
    
    user_query = st.text_input("Ask the question")
    
    if user_query:
        user_queries.append(user_query)
        st.session_state.user_queries += 1
        
        relevant_passages = retrieve_passages(user_query, [pdf_text], vectorizer, tfidf_matrix)

        st.subheader("Relevant response")
        
        response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Context: {relevant_passages}\n\nQuestion: {user_query}"}
                    ]
                )
        answer = response.choices[0].message.content.strip()
        st.write("Answer:", answer)
            