import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Optional
import re
import pickle
from pathlib import Path
import os
from datetime import datetime


class ConversationalLoanRAG:
    def __init__(self, 
                 data_file: str,
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_gpu: bool = False,
                 cache_dir: str = 'cache',
                 use_llm: bool = True,
                 llm_provider: str = 'ollama',
                 llm_model: str = 'llama2'):
        """
        Conversational RAG with FREE LLM generation and persistent embeddings
        
        Args:
            data_file: Path to scraped JSON data
            model_name: HuggingFace model for embeddings
            use_gpu: Use GPU for FAISS
            cache_dir: Directory to store embeddings and index
            use_llm: Whether to use LLM for generation (True) or rule-based (False)
            llm_provider: 'ollama' (FREE, local) or 'huggingface' (FREE, API)
            llm_model: Model name (e.g., 'llama2', 'mistral', 'phi-2')
        """
        self.data_file = data_file
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_client = None
        
        
        self.conversation_history = []
        
        # Initialize components
        print("[INIT] Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
       
        if self.use_llm:
            self._init_llm()
        
        
        self.documents = None
        self.index = None
        self.load_or_build_index()
        
        print(f"[READY] RAG system with {len(self.documents)} documents")
    
    def _init_llm(self):
        """Initialize FREE LLM"""
        print(f"[LLM] Initializing {self.llm_provider} with {self.llm_model}...")
        
        if self.llm_provider == 'ollama':
            try:
                import ollama
                self.llm_client = ollama
                
                
                try:
                    ollama.list()
                    print(f"[LLM]  Ollama is running")
                except:
                    print(f"[LLM]   Ollama not running. Start with: ollama serve")
                    print(f"[LLM]   Falling back to rule-based generation")
                    self.use_llm = False
                    
            except ImportError:
                print("[LLM]  Ollama not installed. Install: pip install ollama")
                print("[LLM]   Falling back to rule-based generation")
                self.use_llm = False
        
        elif self.llm_provider == 'huggingface':
            try:
                from transformers import pipeline
                print(f"[LLM] Loading {self.llm_model} from Hugging Face...")
                self.llm_client = pipeline(
                    "text-generation",
                    model=self.llm_model,
                    max_new_tokens=512,
                    device=0 if self.use_gpu else -1
                )
                print(f"[LLM] ✅ Hugging Face model loaded")
            except ImportError:
                print("[LLM]  Transformers not installed. Install: pip install transformers")
                print("[LLM]   Falling back to rule-based generation")
                self.use_llm = False
            except Exception as e:
                print(f"[LLM]  Error loading model: {e}")
                print("[LLM]   Falling back to rule-based generation")
                self.use_llm = False
        
        else:
            print(f"[LLM]  Unknown provider: {self.llm_provider}")
            self.use_llm = False
    
    def get_cache_paths(self):
        """Get paths for cached files"""
        # Use filename only for stable hashing
        filename = Path(self.data_file).name
        data_hash = str(abs(hash(filename)))[-8:]
        
        return {
            'documents': self.cache_dir / f'documents_{data_hash}.pkl',
            'embeddings': self.cache_dir / f'embeddings_{data_hash}.npy',
            'index': self.cache_dir / f'index_{data_hash}.faiss'
        }
    
    def load_or_build_index(self):
        """Load cached index or build new one"""
        cache_paths = self.get_cache_paths()
        
        # Check if cache exists
        if (cache_paths['documents'].exists() and 
            cache_paths['embeddings'].exists() and 
            cache_paths['index'].exists()):
            
            print("[CACHE] Loading from cache...")
            self.load_from_cache(cache_paths)
        else:
            print("[BUILD] Building new index...")
            self.build_and_save_index(cache_paths)
    
    def load_from_cache(self, cache_paths):
        """Load documents and index from cache"""
        # Load documents
        with open(cache_paths['documents'], 'rb') as f:
            self.documents = pickle.load(f)
        print(f"[CACHE] Loaded {len(self.documents)} documents")
        
        # Load FAISS index
        self.index = faiss.read_index(str(cache_paths['index']))
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        print(f"[CACHE] Loaded FAISS index with {self.index.ntotal} vectors")
    
    def build_and_save_index(self, cache_paths):
        """Build new index and save to cache"""
        # Load and clean documents
        self.documents = self.load_and_clean_data(self.data_file)
        
        # Build embeddings
        texts = [doc['text'] for doc in self.documents]
        print(f"[BUILD] Encoding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("[BUILD] Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
            gpu_index.add(embeddings.astype('float32'))
            # Copy back to CPU for saving
            self.index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            self.index.add(embeddings.astype('float32'))
        
        # Save to cache
        print("[CACHE] Saving to cache...")
        with open(cache_paths['documents'], 'wb') as f:
            pickle.dump(self.documents, f)
        
        np.save(cache_paths['embeddings'], embeddings)
        faiss.write_index(self.index, str(cache_paths['index']))
        
        print(f"[CACHE] Saved to {self.cache_dir}")
    
    def load_and_clean_data(self, data_file: str) -> List[Dict]:
        """Load and clean scraped data"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned = []
        seen_texts = set()
        
        for doc in data:
            text = doc.get('text', '').strip()
            
            if not text or len(text) < 20:
                continue
            
            if text in seen_texts:
                continue
            
            seen_texts.add(text)
            cleaned.append(doc)
        
        print(f"[INFO] Cleaned {len(data)} → {len(cleaned)} documents")
        return cleaned
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms"""
        query_lower = query.lower()
        
        expansions = {
            r'\bhome\s+loan\b': 'home loan housing loan property loan mortgage',
            r'\bcar\s+loan\b': 'car loan vehicle loan automobile loan auto loan',
            r'\bpersonal\s+loan\b': 'personal loan unsecured loan consumer loan',
            r'\beducation\s+loan\b': 'education loan student loan study loan',
            r'\bgold\s+loan\b': 'gold loan loan against gold jewel loan',
            r'\binterest\s+rate\b': 'interest rate ROI rate of interest APR',
            r'\bemi\b': 'EMI equated monthly installment monthly payment repayment',
            r'\bprocessing\s+fee': 'processing fee processing charge administrative fee',
            r'\beligibility\b': 'eligibility criteria eligible requirements qualification',
            r'\bdocument': 'document documentation papers proof certificate',
            r'\btenure\b': 'tenure duration period term loan period',
        }
        
        expanded = query
        for pattern, expansion in expansions.items():
            if re.search(pattern, query_lower):
                expanded += f" {expansion}"
        
        return expanded
    
    def extract_filters(self, query: str) -> Dict[str, Optional[str]]:
        """Extract loan type and section filters"""
        query_lower = query.lower()
        
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
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search with FAISS"""
        expanded_query = self.expand_query(query)
        
        query_embedding = self.model.encode(
            [expanded_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        search_k = min(top_k * 5, len(self.documents))
        distances, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            doc = self.documents[idx]
            
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
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate response"""
        try:
            if self.llm_provider == 'ollama':
                response = self.llm_client.chat(
                    model=self.llm_model,
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are a helpful banking assistant for Bank of Maharashtra. Answer questions accurately based on the provided context.'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    options={
                        'temperature': 0.3,
                        'num_predict': 400,
                    }
                )
                return response['message']['content']
            
            elif self.llm_provider == 'huggingface':
                result = self.llm_client(
                    prompt,
                    max_new_tokens=400,
                    temperature=0.3,
                    do_sample=True,
                )[0]['generated_text']
                
                # Extract only the new generated part
                if prompt in result:
                    result = result.replace(prompt, '').strip()
                
                return result
        
        except Exception as e:
            print(f"[LLM] Error generating response: {e}")
            return None
    
    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """
        Generate answer using LLM or rule-based approach
        """
        if not context_docs:
            return "I couldn't find relevant information about your query. Please try rephrasing your question or ask about specific loan types."
        
        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs[:5], 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Loan: {doc['loan_name']}\n"
                f"Section: {doc['section']}\n"
                f"Information: {doc['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Try LLM generation if enabled
        if self.use_llm and self.llm_client:
            # Get conversation history
            history_context = ""
            if self.conversation_history:
                recent = self.conversation_history[-2:]
                hist_parts = []
                for turn in recent:
                    hist_parts.append(f"User: {turn['query']}")
                    hist_parts.append(f"Assistant: {turn['answer'][:100]}...")
                history_context = "\n".join(hist_parts)
            
            # Create prompt
            prompt = f"""You are a helpful banking assistant for Bank of Maharashtra loans. Answer the user's question based ONLY on the provided context.

CONVERSATION HISTORY:
{history_context if history_context else "No previous conversation."}

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the context provided above
2. Be specific and mention the loan type and section
3. If the context doesn't fully answer the question, say so
4. Use a professional but friendly tone
5. Keep the answer concise (3-4 sentences)
6. Do NOT make up information

ANSWER:"""
            
            llm_response = self._call_llm(prompt)
            
            if llm_response:
                # Add source reference
                top_doc = context_docs[0]
                return f"""{llm_response}

**Source**: {top_doc['loan_name']} - {top_doc['section']}
**Reference**: {top_doc['source_url']}"""
        
        # Fallback to rule-based generation
        top_doc = context_docs[0]
        context_texts = [doc['text'] for doc in context_docs[:3]]
        combined_context = "\n\n".join(context_texts)
        
        return f"""Based on the information from Bank of Maharashtra:

{combined_context}

**Source**: {top_doc['loan_name']} - {top_doc['section']}
**Relevance Score**: {top_doc['similarity_score']:.2f}

For complete details, please visit: {top_doc['source_url']}"""
    
    def get_conversation_context(self, max_turns: int = 3) -> str:
        """Get recent conversation history for context"""
        if not self.conversation_history:
            return ""
        
        recent = self.conversation_history[-max_turns:]
        context_parts = []
        
        for turn in recent:
            context_parts.append(f"User: {turn['query']}")
            context_parts.append(f"Assistant: {turn['answer'][:200]}...")
        
        return "\n".join(context_parts)
    
    def query(self, question: str, top_k: int = 3, use_history: bool = True) -> Dict:
        """
        Main query interface with conversation history
        """
        print(f"\n{'='*80}")
        print(f"[QUERY] {question}")
        print(f"{'='*80}")
        
        # Extract filters from current question only
        filters = self.extract_filters(question)
        
        if filters['loan_type']:
            print(f"[FILTER] Detected loan type: {filters['loan_type']}")
        if filters['section']:
            print(f"[FILTER] Detected section: {filters['section']}")
        
        # For retrieval, use current question + minimal context
        search_query = question
        
        # Only add loan type context from previous query if current query is ambiguous
        if use_history and self.conversation_history and not filters['loan_type']:
            last_turn = self.conversation_history[-1]
            last_filters = self.extract_filters(last_turn['query'])
            if last_filters.get('loan_type'):
                search_query = f"{last_filters['loan_type']} loan: {question}"
                print(f"[CONTEXT] Using context from previous query: {last_filters['loan_type']} loan")
        
        # Search
        results = self.search(search_query, top_k=top_k, filters=filters)
        
        if not results:
            print("[RETRY] No results with filters, searching without filters...")
            results = self.search(search_query, top_k=top_k)
        
        if not results:
            answer = "I couldn't find relevant information. Please try rephrasing your question."
            metadata = {'filters': filters, 'results_found': 0}
        else:
            # Generate answer
            answer = self.generate_answer(question, results)
            metadata = {
                'filters': filters,
                'results_found': len(results),
                'top_loan': results[0]['loan_name'],
                'top_section': results[0]['section'],
                'timestamp': datetime.now().isoformat(),
                'used_llm': self.use_llm
            }
        
        # Store in conversation history
        self.conversation_history.append({
            'query': question,
            'answer': answer,
            'sources': results,
            'metadata': metadata
        })
        
        print(f"\n[RESULTS] Found {len(results)} relevant documents")
        print(f"[GENERATION] {'LLM' if self.use_llm else 'Rule-based'}")
        print(f"{'='*80}\n")
        
        return {
            'answer': answer,
            'sources': results,
            'metadata': metadata
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("[INFO] Conversation history cleared")
    
    def get_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.conversation_history
    
    def save_conversation(self, filename: str):
        """Save conversation history to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Conversation history → {filename}")


def interactive_chat(rag: ConversationalLoanRAG):
    """Interactive chat interface"""
    print("\n" + "="*80)
    print("BANK OF MAHARASHTRA LOAN ASSISTANT")
    print("="*80)
    print(f"\n🤖 Mode: {'LLM Generation' if rag.use_llm else 'Rule-based Generation'}")
    if rag.use_llm:
        print(f"🧠 LLM: {rag.llm_provider} - {rag.llm_model}")
    print("\nCommands:")
    print("  - Type your question to get answers")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'history' to see conversation history")
    print("  - Type 'quit' or 'exit' to end")
    print("="*80 + "\n")
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thank you for using Bank of Maharashtra Loan Assistant.")
                break
            
            if user_input.lower() == 'clear':
                rag.clear_history()
                continue
            
            if user_input.lower() == 'history':
                history = rag.get_history()
                print(f"\n📜 Conversation History ({len(history)} turns):")
                for i, turn in enumerate(history, 1):
                    print(f"\n{i}. Q: {turn['query']}")
                    print(f"   A: {turn['answer'][:150]}...")
                continue
            
            # Query RAG
            result = rag.query(user_input, top_k=3)
            
            # Display answer
            print(f"\n🤖 Assistant:\n")
            print(result['answer'])
            
            # Display sources
            if result['sources']:
                print(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'][:2], 1):
                    print(f"   {i}. {source['loan_name']} - {source['section']} (Score: {source['similarity_score']:.2f})")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


# Main execution
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Conversational RAG with FREE LLM')
    parser.add_argument('--no-llm', action='store_true', help='Use rule-based generation instead of LLM')
    parser.add_argument('--provider', default='ollama', choices=['ollama', 'huggingface'],
                       help='LLM provider (default: ollama)')
    parser.add_argument('--model', default='gpt-oss:20b-cloud',
                       help='Model name (default: gpt-oss:20b-cloud for ollama, microsoft/phi-2 for huggingface)')
    parser.add_argument('--test', action='store_true', help='Run test queries')
    parser.add_argument('--data-file', help='Path to data file')
    
    args = parser.parse_args()
    
    # Get data file path
    script_dir = Path(__file__).parent
    
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        data_file = script_dir.parent / 'data' / 'loan_data_chunked.json'
        cleaned_file = script_dir.parent / 'data' / 'loan_data_chunked_cleaned.json'
        
        if cleaned_file.exists():
            data_file = cleaned_file
            print(f"[INFO] ✅ Using cleaned data: {cleaned_file.name}")
        elif data_file.exists():
            print(f"[INFO] ⚠️  Using raw data: {data_file.name}")
        else:
            print(f"❌ Error: No data file found!")
            print(f"   Run 'python scraper.py' first")
            sys.exit(1)
    
    # Set default model for huggingface
    llm_model = args.model
    if args.provider == 'huggingface' and args.model == 'llama2':
        llm_model = 'microsoft/phi-2'  # Small, fast, free model
    
    # Initialize RAG
    rag = ConversationalLoanRAG(
        data_file=str(data_file),
        model_name='all-MiniLM-L6-v2',
        use_gpu=False,
        cache_dir='cache',
        use_llm=not args.no_llm,
        llm_provider=args.provider,
        llm_model=llm_model
    )
    
    # Run based on mode
    if args.test:
        # Test mode
        test_queries = [
            "What is the interest rate for home loan?",
            "What documents are required?",
            "What about car loan?",
            "How do I apply?",
            "What is the eligibility for personal loan?"
        ]
        
        print("\n" + "="*80)
        print("TESTING CONVERSATIONAL RAG WITH LLM")
        print("="*80)
        
        for query in test_queries:
            result = rag.query(query, top_k=2)
            print(f"\n{result['answer']}\n")
            print("-"*80)
    else:
        # Interactive mode
        interactive_chat(rag)