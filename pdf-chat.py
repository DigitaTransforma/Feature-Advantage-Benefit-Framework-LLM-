import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import find_dotenv, load_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import time

# Load the HuggingFaceHub API token from the .env file
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Load the LLM model from the HuggingFaceHub
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"  # "HuggingFaceH4/zephyr-7b-beta"
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)

# Create a Streamlit app
st.title("Ask Questions About a PDF 📄")

st.sidebar.header("PDF Upload")
pdf_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

upload_message = st.sidebar.empty()
generate_message = st.sidebar.empty()

if pdf_file is not None:
    upload_message.text("Document upload is in progress...")
    upload_message.progress(0)

    # Extract text from the PDF file using PdfReader
    pdf_text = ""
    pdf_reader = PdfReader(pdf_file)
    total_pages = len(pdf_reader.pages)

    for i, page in enumerate(pdf_reader.pages):
        pdf_text += page.extract_text()
        progress = (i + 1) / total_pages
        upload_message.progress(progress)
        time.sleep(0.1)  # Add a slight delay to update the progress bar

    upload_message.text("Document upload is complete")
    upload_message.success("Document upload is complete")

user_question = st.sidebar.text_area("Ask a question about the PDF: ")

if st.sidebar.button("Get Answer"):
    if pdf_file is not None and user_question.strip() != "":
        with generate_message:
            st.text("Generating answer...")  # Show the spinner

        # Create a dictionary with the question as a variable
        prompt_dict = {"question": user_question, "pdf_text": pdf_text}

        # Create a PromptTemplate and LLMChain
        template = "Document: {pdf_text}\n\nQuestion: {question}\n\nAnswer: Let's think step by step."
        prompt = PromptTemplate(template=template, input_variables=["question", "pdf_text"])
        llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

        # Run the LLMChain
        response = llm_chain.run(prompt_dict)

        generate_message.empty()  # Remove the spinner

        st.title("Response")
        st.write(response)  # Display the response
