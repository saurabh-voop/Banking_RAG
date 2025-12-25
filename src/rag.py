import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Optional
import re
from pathlib import Path


class AdvancedLoanRAG:
    def __init__(self, 
                 data_file: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_gpu: bool = False):
        """
        Advanced RAG with FAISS indexing and query expansion
        
        Args:
            data_file: Path to scraped JSON data
            model_name: HuggingFace model for embeddings
            use_gpu: Use GPU for FAISS (if available)
        """
        print("[INIT] Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print("[INIT] Loading documents...")
        self.documents = self.load_and_clean_data(data_file)
        
        print("[INIT] Building FAISS index...")
        self.index = None
        self.use_gpu = use_gpu
        self.build_faiss_index()
        
        print(f"[READY] RAG system initialized with {len(self.documents)} documents")
    
    def load_and_clean_data(self, data_file: str) -> List[Dict]:
        """Load and clean scraped data"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned = []
        seen_texts = set()
        
        for doc in data:
            text = doc.get('text', '').strip()
            
            # Skip invalid documents
            if not text or len(text) < 20:
                continue
            
            # Skip duplicates
            if text in seen_texts:
                continue
            
            seen_texts.add(text)
            cleaned.append(doc)
        
        print(f"[INFO] Cleaned {len(data)} → {len(cleaned)} documents (removed duplicates)")
        return cleaned
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        texts = [doc['text'] for doc in self.documents]
        
        print(f"[INFO] Encoding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine for normalized
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("[INFO] Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index.add(embeddings.astype('float32'))
        print(f"[INFO] FAISS index built: {self.index.ntotal} vectors")
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms
        """
        query_lower = query.lower()
        
        # Domain-specific expansions
        expansions = {
            # Loan types
            r'\bhome\s+loan\b': 'home loan housing loan property loan mortgage',
            r'\bcar\s+loan\b': 'car loan vehicle loan automobile loan auto loan',
            r'\bpersonal\s+loan\b': 'personal loan unsecured loan consumer loan',
            r'\beducation\s+loan\b': 'education loan student loan study loan',
            r'\bgold\s+loan\b': 'gold loan loan against gold jewel loan',
            
            # Financial terms
            r'\binterest\s+rate\b': 'interest rate ROI rate of interest APR',
            r'\bemi\b': 'EMI equated monthly installment monthly payment repayment',
            r'\bprocessing\s+fee': 'processing fee processing charge administrative fee',
            r'\beligibility\b': 'eligibility criteria eligible requirements qualification',
            r'\bdocument': 'document documentation papers proof certificate',
            r'\btenure\b': 'tenure duration period term loan period',
            r'\bmargin\b': 'margin down payment contribution equity',
            r'\bcollateral\b': 'collateral security mortgage pledge guarantee',
            
            # Actions
            r'\bhow\s+to\s+apply\b': 'how to apply application process apply online application procedure',
            r'\brequired\b': 'required needed necessary mandatory essential',
        }
        
        expanded = query
        for pattern, expansion in expansions.items():
            if re.search(pattern, query_lower):
                expanded += f" {expansion}"
        
        return expanded
    
    def extract_filters(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract loan type and section filters from query
        """
        query_lower = query.lower()
        
        # Detect loan type
        loan_type = None
        loan_patterns = {
            'home': r'\b(home|housing|property|mortgage)\s+(loan)?\b',
            'car': r'\b(car|vehicle|auto|automobile)\s+(loan)?\b',
            'personal': r'\bpersonal\s+(loan)?\b',
            'education': r'\b(education|student|study)\s+(loan)?\b',
            'gold': r'\bgold\s+(loan)?\b',
            'two wheeler': r'\b(two\s*wheeler|bike|motorcycle|scooter)\s+(loan)?\b',
        }
        
        for loan, pattern in loan_patterns.items():
            if re.search(pattern, query_lower):
                loan_type = loan
                break
        
        # Detect section
        section = None
        section_patterns = {
            'Interest Rate': r'\b(interest|rate|roi|apr)\b',
            'Eligibility': r'\b(eligibility|eligible|qualify|criteria|requirement)\b',
            'Documents Required': r'\b(document|documentation|paper|proof|certificate)\b',
            'Processing Fees': r'\b(processing\s+fee|charge|cost)\b',
            'Loan Tenure': r'\b(tenure|duration|period|term)\b',
            'Repayment': r'\b(emi|repayment|payment|installment)\b',
            'How to Apply': r'\b(apply|application|process|procedure)\b',
        }
        
        for sec, pattern in section_patterns.items():
            if re.search(pattern, query_lower):
                section = sec
                break
        
        return {'loan_type': loan_type, 'section': section}
    
    def search(self,
               query: str,
               top_k: int = 5,
               filters: Optional[Dict] = None) -> List[Dict]:
        """
        Search with FAISS and apply filters
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Dict with 'loan_type' and 'section' keys
        
        Returns:
            List of relevant documents with scores
        """
        # Expand query
        expanded_query = self.expand_query(query)
        
        # Encode query
        query_embedding = self.model.encode(
            [expanded_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search more candidates for filtering
        search_k = min(top_k * 5, len(self.documents))
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Collect and filter results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc = self.documents[idx]
            
            # Apply filters
            if filters:
                if filters.get('loan_type'):
                    if filters['loan_type'].lower() not in doc['loan_name'].lower():
                        continue
                
                if filters.get('section'):
                    if filters['section'].lower() not in doc['section'].lower():
                        continue
            
            results.append({
                'text': doc['text'],
                'loan_name': doc['loan_name'],
                'section': doc['section'],
                'source_url': doc['source_url'],
                'similarity_score': float(dist),
                'content_type': doc.get('content_type', 'unknown'),
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> Dict:
        """
        Main query interface with automatic filter extraction
        
        Returns:
            Dict with 'answer', 'sources', and 'metadata'
        """
        # Extract filters from query
        filters = self.extract_filters(question)
        
        if verbose:
            print(f"\n[QUERY] {question}")
            if filters['loan_type']:
                print(f"[FILTER] Detected loan type: {filters['loan_type']}")
            if filters['section']:
                print(f"[FILTER] Detected section: {filters['section']}")
        
        # Search with filters
        results = self.search(question, top_k=top_k, filters=filters)
        
        if not results:
            # Retry without filters
            if verbose:
                print("[RETRY] No results with filters, searching without filters...")
            results = self.search(question, top_k=top_k)
        
        if not results:
            return {
                'answer': "I couldn't find relevant information. Please try rephrasing your question.",
                'sources': [],
                'metadata': {'filters': filters, 'results_found': 0}
            }
        
        # Format answer
        answer_parts = []
        for i, result in enumerate(results, 1):
            answer_parts.append(
                f"**{result['loan_name']} - {result['section']}**\n"
                f"{result['text']}\n"
                f"*(Relevance: {result['similarity_score']:.2f})*"
            )
        
        answer = "\n\n".join(answer_parts)
        
        return {
            'answer': answer,
            'sources': results,
            'metadata': {
                'filters': filters,
                'results_found': len(results),
                'top_loan': results[0]['loan_name'],
                'top_section': results[0]['section']
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize
    # Get the path relative to the script location
    script_dir = Path(__file__).parent
    data_file = script_dir.parent / 'data' / 'loan_data_chunked_cleaned.json'
    rag = AdvancedLoanRAG(str(data_file))
    
    # Test queries
    test_queries = [
        "What is the interest rate for home loan?",
        "What documents do I need for a car loan?",
        "Am I eligible for a personal loan?",
        "What is the processing fee for gold loan?",
        "How long is the tenure for education loan?",
        "How can I apply for a home loan online?",
        "What is the EMI for car loan?",
    ]
    
    print("\n" + "="*80)
    print("TESTING ADVANCED RAG SYSTEM")
    print("="*80)
    
    for query in test_queries:
        result = rag.query(query, top_k=2, verbose=True)
        
        print("\n" + "-"*80)
        print("ANSWER:")
        print(result['answer'])
        print("\nMETADATA:", result['metadata'])
        print("="*80 + "\n")