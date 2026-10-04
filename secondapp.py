from langchain_aws import BedrockEmbeddings,ChatBedrock
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains import retrieval_qa
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
import boto3
from dotenv import load_dotenv

st.set_page_config('aws🗿',layout='wide')
st.title('ChatBedRock🗿')
st.sidebar.title('Menu')

bedrock=boto3.client('bedrock-runtime')

def DataIngestion():
    if 'db' not in st.session_state:
      docs=PyPDFDirectoryLoader('documents')
      doc=docs.load()
      splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=2000)
      splits=splitter.split_documents(doc)
      embedder=BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0')
      vector_store=FAISS.from_documents(splits,embedder)
      st.session_state.db=vector_store.as_retriever()
  

if st.sidebar.button("embed"):
    with st.sidebar.spinner('embedding....'):
      DataIngestion()
    
query=st.text_input('ask ur query')

def getAmazon():
    llm=ChatBedrock(model='amazon.nova-lite-v1:0',client=bedrock)
    return llm
def getMistral():
     llm=ChatBedrock(model='mistral.mistral-7b-instruct-v0:2',client=bedrock)
     return llm

prompt=ChatPromptTemplate.from_messages(
    [('system','''you are a helpful research assisstant. 
                do not hallucinate and only use this as knowledgebase.
                <context>
                {context}
                <context>'''),
      ('user','{input}')]
  )  

def getResponse(llm,db,query):
    stuff=create_stuff_documents_chain(llm,prompt)
    bigChain=create_retrieval_chain(db,stuff)
    response=bigChain.invoke({'input':query})
    return response



if st.button('AmazonNova'):
    llm=getAmazon()
    res=getResponse(llm,st.session_state.db,query)
    st.write(res['answer'])  

if st.button('mistral'):
    llm=getMistral()
    res=getResponse(llm,st.session_state.db,query)  
    st.write(res['answer'])  
    
    
