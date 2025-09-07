import os
from neo4j import GraphDatabase


def get_driver():
    """
    Create and return a Neo4j driver from environment variables.
    """
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri=uri, auth=(user, password))
    return driver

def create_fulltext_index(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)


def create_index():
    """
    Creates a fulltext index on entity nodes. Safe to wrap in try/except.
    """
    driver = get_driver()
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
    driver.close()


def get_edges(tx):
    """
    Return all directed edges in the format (source_id, target_id).
    """
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.id AS source, m.id AS target
    """
    result = tx.run(query)
    return [(record["source"], record["target"]) for record in result]


def fetch_all_edges():
    """
    Helper that opens a session, fetches edges, and returns them.
    """
    driver = get_driver()
    with driver.session() as session:
        edges = session.read_transaction(get_edges)
    driver.close()
    return edges


def get_node_text(tx, node_id: str) -> str:
    """
    Retrieves the 'text' property from a :Document node in Neo4j.
    Adjust if your schema differs (property name, label, etc.).
    """
    query = """
    MATCH (d:Document {id: $node_id})
    RETURN d.text AS text
    LIMIT 1
    """
    record = tx.run(query, node_id=node_id).single()
    return record["text"] if record and record["text"] else ""


def fetch_node_texts(node_ids):
    """
    For a list of node_ids, fetch the text for each from Neo4j.
    Returns a dict { node_id: text }
    """
    driver = get_driver()
    node_texts = {}
    with driver.session() as session:
        for nid in node_ids:
            txt = session.read_transaction(get_node_text, nid)
            node_texts[nid] = txt
    driver.close()
    return node_texts
