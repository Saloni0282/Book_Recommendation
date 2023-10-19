import os
import sys
from flask import Flask, request, jsonify
from dotenv import load_dotenv, find_dotenv
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import subprocess
from flask_cors import CORS
import json
import pinecone
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from collections import Counter
import re


app = Flask(__name__)
CORS(app)

load_dotenv(find_dotenv())
OPENAI_API_KEY = "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"

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

# Initialize Pinecone vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "cloozo"
docsearch = Pinecone.from_texts([], embeddings, index_name=index_name)

# Initialize conversation history list
conversation_history = []

@app.route('/chat', methods=['POST'])
def get_answer():
    data = request.get_json()
    query = data.get('query', '')

    # Store user query in conversation history
    conversation_history.append({'role': 'user', 'message': query})

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)

    # Generate suggestions based on conversation history
    suggestions = generate_suggestions(conversation_history)

    response = {
        'query': query,
        'answer': answer,
        'suggestions': suggestions
    }

    return jsonify(response)

# Function to generate suggestions based on conversation history
def generate_suggestions(conversation_history):
    # Extract user queries from conversation history
    user_queries = [entry['message'] for entry in conversation_history if entry['role'] == 'user']

    # Tokenize and count words in user queries
    words = re.findall(r'\b\w+\b', ' '.join(user_queries).lower())
    word_counts = Counter(words)

    # Generate suggestions based on frequently occurring words
    suggestions = []
    for word, count in word_counts.most_common():
        # Generate suggestions using frequently occurring words
        suggestion = f"Tell me more about {word.capitalize()}?"
        suggestions.append(suggestion)

    # Duplicate suggestions and limit the number of suggestions
    unique_suggestions = list(set(suggestions))[:5]  # Limit to 5 unique suggestions

    return unique_suggestions

if __name__ == "__main__":
    app.run(debug=True)

