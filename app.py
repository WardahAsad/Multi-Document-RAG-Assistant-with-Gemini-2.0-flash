# Import required libraries
import streamlit as st  # Main library for creating the web interface
from PyPDF2 import PdfReader  # Library for reading and extracting text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
import os  # For handling environment variables and file operations
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For creating embeddings using Google's AI
import google.generativeai as genai  # Google's Generative AI library
from langchain.vectorstores import FAISS  # Vector store for efficient similarity search
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat functionality with Google's AI
from langchain.chains.question_answering import load_qa_chain  # For creating question-answering chains
from langchain.prompts import PromptTemplate  # For creating structured prompts
from dotenv import load_dotenv  # For loading environment variables from .env file

# Load environment variables from .env file
load_dotenv()
# Configure Google AI with API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    # Initialize empty string to store combined text
    text = ""
    # Iterate through each PDF file
    for pdf in pdf_docs:
        # Create PDF reader object
        pdf_reader = PdfReader(pdf)
        # Iterate through each page
        for page in pdf_reader.pages:
            # Extract and append text from each page
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Initialize text splitter with chunk size of 10000 and overlap of 1000
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # Split text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Initialize embeddings using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create FAISS vector store from text chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save vector store locally
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Define the prompt template for the AI
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize the chat model with specific parameters
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0.4)  # Lower temperature for more focused responses

    # Create prompt template with defined variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Create and return the QA chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the saved vector store with security parameter
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Find relevant documents for the question
    docs = new_db.similarity_search(user_question)

    # Get the QA chain
    chain = get_conversational_chain()
    
    # Generate response using the chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)

    # Print and display the response
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    # Configure the Streamlit page
    st.set_page_config("Chat PDF")
    st.header("Chat with PDFs using Gemini💁")

    # Create text input for user questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Process question if user has entered one
    if user_question:
        user_input(user_question)

    # Create sidebar for file upload
    with st.sidebar:

        st.markdown(""" 
                    <style>
                    .spacer {
                    margin-bottom: 15px;
                }
            </style>
            """, unsafe_allow_html=True)

        st.title("Control Panel")

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # Create file uploader for PDFs
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # Create process button
        if st.button("✅ Submit & Process"):
            with st.spinner("⏳ Processing..."):  # Show loading spinner
                # Process uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done")  # Show success message

# Entry point of the application
if __name__ == "__main__":
    main()