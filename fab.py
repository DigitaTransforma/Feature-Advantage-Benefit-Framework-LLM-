import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

# Load the HuggingFaceHub API token from the .env file
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# Load the Mistral LLM model from the HuggingFaceHub
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"  # "HuggingFaceH4/zephyr-7b-beta"
mistral_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
)

# Define the Prompt Template for FAB framework with an instruction
template = """Instruction: You are an expert sales leader that specializes 
in using the feature-advantage-benefit framework.
Generate 10 FAB (Features, advantage benefit) statements in a table.

Product: {product}
Audience: {audience}

"""

prompt = PromptTemplate(template=template, input_variables=["product", "audience"])

# Create a Streamlit app
st.title("FAB Framework Generator")
st.text("Feature-Advantage-Benefit (FAB) Framework Generator for marketing purpose")
st.text("- Built on Mistral 7B LLM")

# User input: Text field for the product/business and text area for context
product_input = st.text_input("Define your Business or Product:")
audience_input = st.text_area("Define your Audience:")

# Button to trigger FAB generation
if st.button("Generate FAB"):
    # Check if required fields are empty
    if not product_input or not audience_input:
        st.error("Please fill in both the product/business and audience fields.")
    else:
        # Display a spinner while generating the FAB statements
        with st.spinner("Generating FAB statements..."):
            # Create a dictionary with user inputs
            user_input_dict = {"product": product_input, "audience": audience_input}

            # Create an LLMChain for FAB generation using the prompt template
            llm_chain = LLMChain(prompt=prompt, llm=mistral_llm)

            # Run the LLMChain for FAB generation
            fab_response = llm_chain.run(user_input_dict)

        # Display the FAB statements after the spinner
        st.write("Here are 10 FAB Statements that resonates with your audience:")
        st.write(fab_response)
