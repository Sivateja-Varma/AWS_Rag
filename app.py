import boto3
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings,ChatBedrock
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import os 
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config('awsRag🗿',layout='wide')
st.title('RAG with aws-bedrock 🗿')

st.sidebar.title('Menu')
files=st.sidebar.file_uploader('vectorStore',type='pdf')
input=st.text_input('enter your query: ')

if files:
  file_path=F'./{files.name}'
  with open(file_path,'wb+') as f:
    f.write(files.getvalue())
  if 'vector' not in st.session_state:
    st.session_state.loader=PyPDFLoader(file_path)
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.textsplitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.splits=st.session_state.textsplitter.split_documents(st.session_state.docs)
    st.session_state.embedder=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
    st.session_state.vector=FAISS.from_documents(st.session_state.splits,st.session_state.embedder)
    st.session_state.retriver=st.session_state.vector.as_retriever()


  prompt=ChatPromptTemplate.from_messages(
    [('system','''you are a helpful research assisstant. 
                do not hallucinate and only use this as knowledgebase.
                <context>
                {context}
                <context>'''),
      ('user','{input}')]
  )
  bedrock=boto3.client('bedrock-runtime')
  model_id='mistral.mistral-7b-instruct-v0:2'
  llm=ChatBedrock(client=bedrock,model=model_id)
  chain=create_stuff_documents_chain(llm,prompt)
  bigchain=create_retrieval_chain(st.session_state.retriver,chain)

  
  if input:
    with st.spinner('thinking...'):
      response=bigchain.invoke({'input':input})
      st.write(response['answer'])


  







