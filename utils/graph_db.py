import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from sklearn.datasets import fetch_20newsgroups
from langchain.schema import Document


def load_data(
    graph_transformer,  # Pass your LLMGraphTransformer from the notebook
    embeddings,         # Pass your OllamaEmbeddings (or any embeddings) from the notebook
    dummy_file_path="dummytext.txt",
    newsgroups_categories=["comp.graphics","sci.med","talk.religion.misc"],
    chunk_size=250,
    chunk_overlap=24
):
    """
    Loads two datasets into Neo4j using existing LLMGraphTransformer & embeddings:
      1) The text from 'dummy_file_path' => label :Document
      2) 20 Newsgroups => label :NewsDoc

    Arguments:
      graph_transformer: an LLMGraphTransformer object (already instantiated in notebook)
      embeddings: an Embeddings object (e.g., OllamaEmbeddings) from notebook
      dummy_file_path: path to your dummy text file
      newsgroups_categories: list of newsgroup categories to fetch
      chunk_size, chunk_overlap: for text chunking
    """

    # 1) Load environment + create a Neo4j driver
    load_dotenv()
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    pwd = os.environ["NEO4J_PASSWORD"]

    driver = GraphDatabase.driver(uri=uri, auth=(user, pwd))

    # 2) Create index if needed
    def create_fulltext_index(tx):
        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_id` 
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        tx.run(query)

    def create_index():
        with driver.session() as session:
            session.execute_write(create_fulltext_index)
            print("Fulltext index created successfully.")

    try:
        create_index()
    except Exception as e:
        print("Index creation issue (possibly already created):", e)

    # 3) Initialize the Neo4jGraph
    graph = Neo4jGraph()  # uses env by default

    # 4) Set up a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    ###############################################
    # PART A: Load dummytext => :Document
    ###############################################
    loader = TextLoader(file_path=dummy_file_path)
    docs_dummy = loader.load()
    docs_dummy_split = text_splitter.split_documents(docs_dummy)

    # Convert via your already-instantiated graph_transformer
    graph_docs_dummy = graph_transformer.convert_to_graph_documents(docs_dummy_split)

    # Add them to Neo4j with label :Document
    graph.add_graph_documents(
        graph_docs_dummy,
        baseEntityLabel=True,   # => label :Document
        include_source=True
    )
    print(f"Loaded {len(graph_docs_dummy)} chunks from {dummy_file_path} into Neo4j (:Document).")

    ###############################################
    # PART B: Load 20 newsgroups => :NewsDoc
    ###############################################
    news_data = fetch_20newsgroups(subset='train', categories=newsgroups_categories)
    raw_texts = news_data.data

    # Create Documents, chunk them
    news_docs = [Document(page_content=txt, metadata={"id": f"news_{i}"})
                 for i, txt in enumerate(raw_texts)]
    news_docs_split = text_splitter.split_documents(news_docs)

    graph_docs_news = graph_transformer.convert_to_graph_documents(news_docs_split)

    # Add them with label :NewsDoc
    graph.add_graph_documents(
        graph_docs_news,
        baseEntityLabel="NewsDoc",
        include_source=True
    )
    print(f"Loaded {len(graph_docs_news)} chunks from 20 Newsgroups into Neo4j (:NewsDoc).")

    ###############################################
    # PART C: Create vector indexes
    ###############################################
    from langchain_community.vectorstores import Neo4jVector

    # For :Document
    vector_index_dummy = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    # For :NewsDoc
    vector_index_news = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="NewsDoc",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    print("Vector indexes created for :Document and :NewsDoc.")

    # 5) Clean up
    driver.close()
    print("All data loaded. Neo4j driver closed.")
