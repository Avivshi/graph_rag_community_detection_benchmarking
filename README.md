# Graph RAG Community Detection Benchmarking

**Author**: Aviv Shimoni  
**Supervisor**: Dr. Uri Itai  
**Institution**: Holon Institute of Technology

---

## 📚 Project Overview

This project implements and benchmarks various community detection algorithms for Graph-enhanced Retrieval-Augmented Generation (Graph RAG) systems, inspired by the methodology from **"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"** by Edge et al. (Microsoft Research, 2024).

We extend their foundational work by comparing **7 different community detection methods** across two classic literature datasets to evaluate their effectiveness in partitioning knowledge graphs for improved RAG performance.

### 🎯 Key Objectives

- **Benchmark Community Detection Algorithms**: Compare 7 different algorithms on graph structures
- **Evaluate RAG Performance**: Test retrieval quality with different graph partitioning strategies  
- **Cross-Dataset Analysis**: Validate findings across Alice in Wonderland and Dr. Jekyll & Mr. Hyde
- **Hallucination Testing**: Assess RAG system robustness against fabricated queries

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.8+** (tested with Python 3.12)
- **Docker** and **Docker Compose**
- **Neo4j** (run via Docker containers)
- **Ollama** with Llama 3.1:8B and MxBai-Embed-Large models

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd graph_rag_community_detection_benchmarking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
# Neo4j Configuration - Alice Dataset
NEO4J_URI=bolt://localhost:7686
NEO4J_USERNAME=neo4j  
NEO4J_PASSWORD=password1

# Alternative configuration for Jekyll dataset
# NEO4J_URI=bolt://localhost:7688
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=password2
```

### 3. Neo4j Database Setup

Start the Neo4j containers using Docker Compose:

```bash
# Start both Neo4j instances
docker-compose up -d

# Verify containers are running
docker ps
```

The setup creates two Neo4j instances:
- **Alice dataset**: `localhost:7686` (password: `password1`)
- **Jekyll dataset**: `localhost:7688` (password: `password2`)

### 4. Ollama Model Setup

Install and start the required Ollama models:

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull required models
ollama pull llama3.1:8b
ollama pull mxbai-embed-large

# Verify models are available
ollama list
```

### 5. Download Datasets

The project includes a utility script to download the literature datasets:

```bash
python utils/download_data.py
```

This downloads:
- **Alice's Adventures in Wonderland** → [`datasets/alice/alice.txt`](datasets/alice/alice.txt)
- **Dr. Jekyll and Mr. Hyde** → [`datasets/jekyll/jekyll.txt`](datasets/jekyll/jekyll.txt)

---

## 🚀 Usage

### Running the Complete Analysis

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**:
   - [`graph_rag_community_detection_benchmarking.ipynb`](graph_rag_community_detection_benchmarking.ipynb)

3. **Execute cells sequentially** to:
   - Ingest datasets into Neo4j
   - Build knowledge graphs
   - Run community detection algorithms
   - Evaluate and compare results

### Quick Start Example

```python
from utils.neo4j_utils import get_driver, fetch_all_edges
from community_detection.community_leiden import leiden_partition

# Activate dataset
activate_dataset("alice")

# Fetch graph data
driver = get_driver()
with driver.session() as session:
    edges = fetch_all_edges()
driver.close()

# Run community detection
leiden_communities = leiden_partition(nodes, edges)
```

---

## 📊 Community Detection Methods

### 1. Structure-Based Methods
- **Fiedler Partition**: Spectral bisection using graph Laplacian
- **Leiden Algorithm**: Modularity optimization with refinement
- **Girvan-Newman**: Edge betweenness-based iterative removal

### 2. Embedding-Based Methods  
- **K-Means Clustering**: Partitioning in semantic vector space
- **Hierarchical Clustering**: Ward linkage on node embeddings
- **Spectral Clustering**: Eigenvalue decomposition on embeddings

### 3. Hybrid Approaches
- **Weighted Leiden**: Combines graph topology + embedding similarity

---

## 📈 Evaluation Metrics

### Graph-Theoretic Metrics
- **Modularity**: Edge density within vs. between communities
- **Conductance**: Edge cut ratio relative to community volume
- **Coverage**: Fraction of edges within communities
- **Performance**: Correctly classified node pairs
- **Normalized Cut**: Balanced partition quality

### Embedding-Based Metrics
- **Silhouette Score**: Cluster cohesion vs. separation
- **Calinski-Harabasz Index**: Cluster dispersion ratio

### Distance Analysis
- **Intra-community Distance**: Semantic similarity within communities
- **Inter-community Distance**: Semantic distance between communities

---

## 📁 Project Structure

```
├── docker-compose.yaml             # Neo4j container configuration
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
├── README.md                       # This file
│
├── datasets/                       # Literature datasets
│   ├── alice/alice.txt
│   └── jekyll/jekyll.txt
│
├── community_detection/            # Algorithm implementations
│   ├── community_fiedler.py
│   ├── community_kmeans.py
│   ├── community_leiden.py
│   ├── community_hierarchical.py
│   ├── community_weighted_leiden.py
│   ├── community_girvan_newman.py
│   └── community_spectral.py
│
├── utils/                          # Utility modules
│   ├── plots.py                    # Visualization functions
│   ├── community_utils.py          # Graph analysis utilities
│   ├── neo4j_utils.py             # Database connection utilities
│   ├── evaluation_metrics.py      # Metric calculation functions
│   └── download_data.py           # Dataset download script
│
├── graph_rag_community_detection_benchmarking.ipynb # Jupyter notebook
│
└── data-*/                         # Neo4j database files (auto-generated)
    ├── data-alice/
    └── data-jekyll/
```

---

## 🔧 Key Dependencies

### Core Libraries
```txt
langchain==0.3.19
langchain-community==0.3.18
langchain-experimental==0.3.4
langchain-ollama==0.2.3
neo4j==5.28.1
networkx==3.4.2
```

### Scientific Computing
```txt
numpy==1.26.4
scipy==1.15.2
scikit-learn==1.6.1
matplotlib==3.10.1
pandas==2.2.3
```

### Graph Analysis
```txt
igraph==0.11.8
python-louvain==0.16
leidenalg==0.10.2
```

See [`requirements.txt`](requirements.txt) for complete dependency list.

---

## 📊 Expected Results

### Performance Benchmarks

| Algorithm | Best Use Case | Key Metrics |
|-----------|---------------|-------------|
| **Leiden** | Balanced RAG performance | Modularity > 0.45, balanced distances |
| **K-Means** | Tight semantic clustering | IntraPW < 0.30, IntraCent < 0.25 |
| **Fiedler** | Maximum separation | Across > 0.70, perfect coverage |
| **Weighted Leiden** | Cross-domain robustness | Consistent across content types |

### Quality Thresholds
- **Excellent performance**: IntraPW ≤ 0.30, Across ≥ 0.65
- **Alert conditions**: IntraPW > 0.45 OR Across < 0.55
- **Re-evaluation trigger**: >15% metric change

---

## 🐛 Troubleshooting

### Common Issues

**Neo4j Connection Errors**:
```bash
# Check container status
docker ps

# Restart containers
docker-compose down && docker-compose up -d

# Check logs
docker-compose logs neo4j-alice
```

**Ollama Model Issues**:
```bash
# Verify models are running
ollama list

# Restart Ollama service
ollama serve
```

**Memory Issues**:
- Reduce batch sizes in embedding generation
- Use smaller `k` values for clustering algorithms
- Monitor system resources during graph analysis

---

## 📚 References

1. Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." Microsoft Research.

2. Blondel, V. D., et al. (2008). "Fast unfolding of communities in large networks." Journal of Statistical Mechanics.

3. Traag, V. A., et al. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." Scientific Reports.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---
