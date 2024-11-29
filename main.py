

from dotenv import load_dotenv

load_dotenv()
app = r"https://www.credibila.com"

import requests
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.schema import Document
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.chroma import Chroma
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

URL = "https://r.jina.ai/https://www.credibila.com/"

LLM = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
LLM_FAST = ChatGroq(model="llama-3.1-8b-instant")

PLACE = "./credibila"

headers = {
    'Authorization' : 'Bearer ' + os.environ['JINA_API_KEY']
}
response = requests.get(url=URL,headers=headers)
print(type(response.text))


if response.status_code==200:
    content=response.text
else:
    raise Exception(f"Failed to retrieve data {response.status_code}")

text_splitters = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    separators=["", " ", '', ' '],
)

splits = text_splitters.split_text(content)
print(len(splits))

# print(type(splits[0])) #str

documents = [Document(page_content=text) for text in splits]
# print(len(documents))
# print(type(documents[0]))

if not os.path.exists(PLACE):
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=PLACE,
    )
# vectorstore.persist()
else:
    vectorstore = Chroma(
        persist_directory=PLACE,
        embedding_function=OpenAIEmbeddings()
    )

retriever = vectorstore.as_retriever()

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = (
    {'context' : retriever, "question": RunnablePassthrough()}
    | prompt
    | LLM_FAST
    | StrOutputParser()
)

while True:
    user_input = input("Hey, Enter your question: ")
    if user_input.lower() in ['quit','exit','nothing','q']:
        break
    else:
        print("BOT: ",chain.invoke(user_input))
