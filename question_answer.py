import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader

# Load the HuggingFaceHub API token from the .env file
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Load the LLM model from the HuggingFaceHub
repo_id = "HuggingFaceH4/zephyr-7b-beta" #"mistralai/Mistral-7B-Instruct-v0.1" #"tiiuae/falcon-7b-instruct"
mistral_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)

# Create a Streamlit app
st.title("AI Question Answering and Summarization")

# User input: Text field for the question
question = st.text_area("Enter your question:", "How do I cook egusi soup?")

# Button to trigger the response
if st.button("Get Answer"):
    # Display a spinner while generating the answer
    with st.spinner("Generating answer..."):
        # Create a dictionary with the question as a variable
        prompt_dict = {"question": question}

        # Create a PromptTemplate and LLMChain
        template = "Question: {question}\n\nAnswer: Let's think step by step."
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=mistral_llm)

        # Run the LLMChain
        response = llm_chain.run(prompt_dict)

    # Display the response after the spinner
    st.write("Response:")
    st.write(response)

# Close the Streamlit app
