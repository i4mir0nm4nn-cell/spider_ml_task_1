
import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

os.environ["GROQ_API_KEY"] = "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"

st.set_page_config(page_title="Research Paper QA", page_icon="📚")
st.title(" 📚Research Paper Assistant")
st.caption("Ask questions from 7 NLP research papers")

@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
    return llm, vectordb

llm, vectordb = load_chain()

prompt_template = """You are an expert AI research assistant.
Use the following context from research papers to answer the question.
Always mention which paper the answer is from.
If the answer is not in the context, say "I do not have enough information from the provided papers."

Context: {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about the research papers..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            docs = vectordb.similarity_search(query, k=4)
            context = "\n\n".join([f"[{d.metadata['paper_name']}]: {d.page_content}" for d in docs])
            final_prompt = prompt.format(context=context, question=query)
            response = llm.invoke(final_prompt)
            sources = list(set([d.metadata['paper_name'] for d in docs]))
            answer = response.content
            full_response = f"{answer}\n\n**Sources:** {', '.join(sources)}"
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
