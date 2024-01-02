from langchain.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


# This method will load the documents/code repo split it into chunks, embed and store it in a vector store locally
def ingestion_docs() -> None:
    # Give the path to the local repository in repo_path and the branch you want to Load
    loader = GitLoader(repo_path="/Users/ksur/papi/desktop-pay-api", branch="master")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    # Another way of splitting based on language specific
    # text_splitter = RecursiveCharacterTextSplitter.from_language(Language.JAVA, chunk_size=400, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} documents")
    embeddings = OpenAIEmbeddings()

    # Vector embeddings will be stored in Local with the given index name. (Same path from where the program is running)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_papi")


if __name__ == '__main__':
    ingestion_docs()
