from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
# import os

st.set_page_config(
    page_title="SiteGPT - Cloudflare Docs",
    page_icon="📡",
)

st.markdown(
    """
    # SiteGPT for Cloudflare Documentation

    Ask questions about Cloudflare's documentation for:
    - AI Gateway
    - Cloudflare Vectorize
    - Workers AI
    
    Provide the sitemap URL and your OpenAI API Key to get started.
    """
)

with st.sidebar:
    openai_api_key = st.text_input("Your OpenAI API Key", type="password")
    st.markdown("https://github.com/oguhn/site-gpt/blob/main/app.py")

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
    """
)

choose_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Use ONLY the following pre-existing answers to answer the user's question.
        Use the answers that have the highest score (more helpful) and favor the most recent ones.
        Cite sources and return the sources of the answers as they are, do not change them.
        Answers: {answers}
        """,
    ),
    ("human", "{question}"),
])

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    llm = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key)
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke({"question": question, "context": doc.page_content}).content,
                "source": doc.metadata.get("source"),
                "date": doc.metadata.get("lastmod"),
            }
            for doc in docs
        ],
    }

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    llm = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key)
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{a['answer']}\nSource:{a['source']}\nDate:{a['date']}" for a in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})

def parse_page(soup):
    for tag in soup(["header", "footer", "nav", "script", "style"]):
        tag.decompose()
    content = soup.get_text(separator=" ")
    return " ".join(content.split())

@st.cache_data(show_spinner="Loading sitemap and embedding content...")
def load_docs_from_sitemap(sitemap_url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)
    loader = SitemapLoader(sitemap_url, parsing_function=parse_page)
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return vector_store.as_retriever()

sitemap_url = st.text_input("Cloudflare Docs Sitemap URL", value="https://developers.cloudflare.com/sitemap.xml")

if openai_api_key and sitemap_url:
    if ".xml" not in sitemap_url:
        st.error("Please provide a valid sitemap.xml URL.")
    else:
        retriever = load_docs_from_sitemap(sitemap_url)
        query = st.text_input("Ask your question about Cloudflare Docs")

        if query:
            chain = (
                {"docs": retriever, "question": RunnablePassthrough()}
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\\$"))
