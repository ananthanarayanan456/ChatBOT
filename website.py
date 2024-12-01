from dotenv import load_dotenv
load_dotenv()
import requests
import os

#LLM
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI

#document processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings

#Vectorstore
from langchain_chroma.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

#prompting
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

#chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

#agents
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START


URL = "https://r.jina.ai/https://www.credibila.com/"

LLM = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
LLM_FAST = ChatGroq(model="llama-3.1-8b-instant")

CHAT_HISTORY=[]

PLACE = "./credibila"

headers = {
    'Authorization' : 'Bearer ' + os.environ['JINA_API_KEY']
}
response = requests.get(url=URL,headers=headers)
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
documents = [Document(page_content=text) for text in splits]

vectorstore=""
if not os.path.exists(PLACE):
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=PLACE,
    )
else:
    vectorstore = Chroma(
        persist_directory=PLACE,
        embedding_function=OpenAIEmbeddings()
    )

retriever = vectorstore.as_retriever(search_kwargs={"k":1})

prompt1 = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
    ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(LLM_FAST,retriever,prompt1)

prompt2 = ChatPromptTemplate.from_messages([
    ("system","Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user","{input}"),
])

document_chain = create_stuff_documents_chain(LLM_FAST,prompt2)
val = create_retrieval_chain(retriever_chain,document_chain)


def answer(ans):
    return val.invoke({
        "chat_history": CHAT_HISTORY,
        "input": ans,
    })['answer']

CHAT_HISTORY.append(SystemMessage(content="Hello, I'm Credibila Bot, How can I help you?"))

while True:
    user_input = input("Hey, Enter your question: ")
    if user_input.lower() in ['quit','exit','nothing','q']:
        CHAT_HISTORY.append(AIMessage(content="Thank You!"))
        break
    else:
        CHAT_HISTORY.append(HumanMessage(content=user_input))
        response_ans = answer(user_input)
        CHAT_HISTORY.append(AIMessage(content=response_ans))
        # print("Chat History :",CHAT_HISTORY)
        print("BOT: ",response_ans)

# for message in CHAT_HISTORY:
#     if isinstance(message,AIMessage):
#         print("BOT :",message.content)