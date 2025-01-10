import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


api_key= "YOUR-OWN-GROQ-API-KEY"

# Define paths and API keys
MODEL_NAME = "llama-3.3-70b-versatile"
vectordb_file_path = "faiss_index"

# Initialize Groq Model
groq_model = ChatGroq(temperature=0.3, groq_api_key=api_key, model_name=MODEL_NAME)
huggingface_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



def load_csv_data(file_paths):
    """Load CSV files into documents."""
    documents = []
    for file_path in file_paths:
        loader = CSVLoader(file_path=file_path)
        data = loader.load()
        documents.extend(data)  # Append data from each file
    return documents

def create_vector_db():
    """Create and save a FAISS vector database."""
    if not os.path.exists(vectordb_file_path):
        os.makedirs(vectordb_file_path)

    file_paths = ['/content/domestic_visitors_updated.csv', '/content/foreign_visitors_updated.csv']  # Update file paths
    data = load_csv_data(file_paths)

    # Check the first few items to ensure the data is loaded correctly
    print("Sample Data:", data[:3])

    # Create a FAISS instance for vector database
    vectordb = FAISS.from_documents(documents=data, embedding=huggingface_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)
    print(f"Vector DB created and saved at {vectordb_file_path}!")

def get_qa_chain():
    """Load the FAISS database and set up a QA chain."""
    if not os.path.exists(vectordb_file_path):
        print(f"Error: FAISS index not found at {vectordb_file_path}. Please run `create_vector_db()` first.")
        return None

    vectordb = FAISS.load_local(vectordb_file_path, huggingface_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Modify the prompt to be compatible with the Groq model
    prompt_template = """
            Given the following context and question, generate an answer based on the provided context first. If the answer is not found, attempt to provide a logical and real-time answer. 

            In your response, try to extract as much relevant information as possible from the provided datasets. Carefully read the question and determine which dataset is the most appropriate for answering. If no direct answer is found, try to give a logical and relevant response based on your understanding.

            If no answer can be derived from the context and you are unable to generate a logical response, please respond with: "I don't know." Do not fabricate an answer.

            Also answers must be to the point not much very long answers. Note you are going to be an assistant of a highly reputed goverment official so reply in a formal manner & as being said short answers are highly valued.

            CONTEXT: {context}

            QUESTION: {question}
            """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Integrate the ChatGroq model for the chain
    chain = RetrievalQA.from_chain_type(
        llm=groq_model,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# Call this function to create the FAISS index if it hasn't been created yet
create_vector_db()



# After the FAISS index is created, you can query using the chain
qa_chain = get_qa_chain()
if qa_chain:
    query = "Cultural / Corporate Events to boost tourism. a. What kind of events the government can conduct? b. Which month(s)? c. Which districts?"

    # Use `invoke` instead of `run` to handle multiple output keys
    result = qa_chain.invoke({"query": query})

    # Extract the result and the source documents
    answer = result['result']
    source_documents = result['source_documents']

    print("Answer:", answer)
    print("Source Documents:", source_documents)
