# import streamlit as st
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub

# load_dotenv() 

# def get_pdf_text(pdf_docs):
#   text = ""
#   for pdf in pdf_docs:
#     pdf_reader = PdfReader(pdf)
#     for page in pdf_reader.pages:
#       text = page.extract_text()
#   return text

# def get_text_chunks(text):
#   text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
#   )
#   chunks = text_splitter.split_text(text)
#   return chunks

# def get_vectorstore(text_chunks):
# #   embeddings = OpenAIEmbeddings()
#   embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={})
#   vectorstore = FAISS.from_text(text=text_chunks, embedding=embeddings)
#   return vectorstore

# def get_conversation_chain(vectorstore):
# #   llm = ChatOpenAI()
#   llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#   memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#   conversation_chain = ConversationalRetrievalChain.from_llm(
#       llm=llm,
#       retriever=vectorstore.as_retriever(),
#       memory=memory
#   )
#   return conversation_chain

# def handle_userinput(user_question):
#   response = st.session_state.conversation({'question': user_question})
#   st.session_state.chat_history = response('chat_history')

#   for i, message in enumerate(st.session_state.chat_history):
#     if i%2 == 0:
#       st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#     else:
#       st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
#    load_dotenv()
#    st.set_page_config(page_title="Multiple-PDF Chatbot", page_icon=":books:")

#    st.write(css, unsafe_allow_html=True)

#    if "conversation" not in st.session_state:
#      st.session_state.conversation = None

#    if "chat_history" not in st.session_state:
#      st.session_state.chat_history = None

#    st.header("Multiple-PDF Chatbot :books:")
#    user_question = st.text_input("Ask a question about your documents:")
#    if user_question:
#      handle_userinput(user_question)


#    with st.sidebar:
#       st.subheader("Your documents")
#       pdf_docs = st.file_uploader(
#          "Upload you PDFs here and click on 'Process'", accept_multiple_files=True)
#       if st.button("Process"):
#         with st.spinner("Processing"):
#         #  get pdf text
#          raw_text = get_pdf_text(pdf_docs)
         
#         # get the text chunks
#          test_chunks = get_text_chunks(raw_text)
       
#         #create vector store
#          vectorstore = get_vectorstore(test_chunks)

#          #create converation chain
#          st.session_state.conversation = get_conversation_chain(vectorstore)


# if __name__ == '__main__':
#    main()

import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from transformers import pipeline

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into clean semantic chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )
    chunks = splitter.split_text(text)
    # Filter short chunks
    chunks = [chunk for chunk in chunks if len(chunk.split()) > 5]
    return chunks

# Create vector store with better embeddings
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embedding=embeddings)

# Use Hugging Face transformer pipeline to generate answer
def generate_answer(prompt):
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    result = generator(prompt, max_new_tokens=512)[0]["generated_text"]
    return result

# Handle user queries and display conversation
def handle_userinput(user_question):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process PDFs first.")
        return

    # Retrieve relevant chunks
    docs = st.session_state.vectorstore.similarity_search(user_question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Prompt crafting
    prompt = f"""Use the following context to answer the question clearly.

Context:
{context}

Question: {user_question}
Answer:"""

    answer = generate_answer(prompt)

    st.session_state.chat_history.append(("You", user_question))
    st.session_state.chat_history.append(("Bot", answer.strip()))

    # Display chat
    for speaker, message in st.session_state.chat_history:
        template = user_template if speaker == "You" else bot_template
        st.write(template.replace("{{MSG}}", message), unsafe_allow_html=True)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Multiple-PDF Chatbot", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Multiple-PDF Chatbot 📚")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Reading and indexing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.success("Documents processed! You can ask questions now.")

if __name__ == "__main__":
    main()
