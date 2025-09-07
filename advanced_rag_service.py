#!/usr/bin/env python3
"""
Advanced RAG Service for Doc-DB
Comprehensive document chunking, vector encoding, ChromaDB integration, and Claude prompting
"""

import os
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import chromadb
import anthropic

# Simple regex-based text processing (no NLTK needed)

# Initialize Flask app
app = Flask(__name__)
# Configure CORS with environment variable support
ALLOWED_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:4200,http://localhost:8080').split(',')
CORS(app, 
     origins=ALLOWED_ORIGINS,
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Initialize ChromaDB client with configurable path
CHROMADB_PATH = os.getenv('CHROMADB_PATH', "/Users/tojojose/trominos/rag-quiz/db/chroma")
chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)

# No embedding model - using ChromaDB's built-in search

# Global Anthropic client (will be initialized later)
anthropic_client = None

def initialize_anthropic_client():
    """Initialize Anthropic client using exact same method as MCP server"""
    global anthropic_client
    
    if anthropic_client:
        return anthropic_client
    
    try:
        # Import MCP server components
        import sys
        MCP_SERVER_PATH = os.getenv('MCP_SERVER_PATH', '/Users/tojojose/trominos/a-tier-mcp-server')
        sys.path.append(MCP_SERVER_PATH)
        
        from api_key_manager import APIKeyManager
        api_key_manager = APIKeyManager()
        print('✅ API Key Manager initialized')
        
        # Get Anthropic API key from database - exactly like MCP server
        key_result = api_key_manager.get_api_key('anthropic')
        if key_result.get('success'):
            api_key = key_result['api_key']
            anthropic_client = anthropic.Anthropic(api_key=api_key)
            print('✅ Anthropic client initialized with database key')
            print(f'   Key: {key_result["key_name"]} (last used: {key_result.get("last_used_at", "never")})')
            return anthropic_client
        else:
            # Fallback to environment variable - exactly like MCP server
            env_key = os.getenv('ANTHROPIC_API_KEY')
            if env_key and env_key not in ['your_anthropic_api_key_here', 'sk-ant-api03-your_key_here']:
                anthropic_client = anthropic.Anthropic(api_key=env_key)
                print('⚠️  Using environment API key (consider migrating to database)')
                return anthropic_client
            else:
                print('❌ No valid Anthropic API key found in database or environment')
                return None
                
    except Exception as e:
        print(f'❌ Client initialization failed: {e}')
        # Try environment fallback - exactly like MCP server
        try:
            env_key = os.getenv('ANTHROPIC_API_KEY')
            if env_key and env_key not in ['your_anthropic_api_key_here']:
                anthropic_client = anthropic.Anthropic(api_key=env_key)
                print('⚠️  Fallback to environment API key')
                return anthropic_client
        except Exception as e2:
            print(f'❌ Environment fallback failed: {e2}')
        
        return None

def get_anthropic_client():
    """Get Anthropic client, initializing if needed"""
    global anthropic_client
    if not anthropic_client:
        anthropic_client = initialize_anthropic_client()
    return anthropic_client

# In-memory storage for training sessions
TRAINING_SESSIONS = {}

# Document chunking classes removed - using existing ChromaDB data

class ChromaVectorStore:
    """ChromaDB vector storage using existing collections and built-in search"""
    
    def __init__(self, collection_name: str = None):
        # Try to find existing collections from your uploaded PDF
        self.collections = self._get_available_collections()
        self.active_collection = self._get_best_collection()
        print(f"📚 Using collection: {self.active_collection.name if self.active_collection else 'None'}")
    
    def _get_available_collections(self):
        """Get all available collections"""
        try:
            collections = chroma_client.list_collections()
            print(f"🔍 Found {len(collections)} collections")
            for col in collections:
                count = col.count()
                print(f"   • {col.name}: {count} documents")
            return collections
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def _get_best_collection(self):
        """Get the collection with the most documents (likely your uploaded PDF)"""
        if not self.collections:
            return None
        
        # Find collection with most documents
        best_collection = max(self.collections, key=lambda c: c.count())
        return best_collection
    
    def search_documents(self, query: str, n_results: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search using ChromaDB's built-in text search"""
        if not self.active_collection:
            return {
                'query': query,
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': [],
                'similarity_scores': []
            }
        
        try:
            query_kwargs = {
                'query_texts': [query],
                'n_results': n_results
            }
            
            if filters:
                query_kwargs['where'] = filters
            
            results = self.active_collection.query(**query_kwargs)
            
            return {
                'query': query,
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'ids': results['ids'][0] if results['ids'] else [],
                'similarity_scores': [1 - dist for dist in results['distances'][0]] if results.get('distances') and results['distances'][0] else []
            }
        except Exception as e:
            print(f"Search error: {e}")
            return {
                'query': query,
                'documents': [],
                'metadatas': [],
                'distances': [],
                'ids': [],
                'similarity_scores': []
            }
    
    def get_collection_info(self):
        """Get collection information"""
        if not self.active_collection:
            return {
                'collection_name': 'None',
                'total_chunks': 0,
                'status': 'No collections found'
            }
        
        try:
            count = self.active_collection.count()
            return {
                'collection_name': self.active_collection.name,
                'total_chunks': count,
                'search_method': 'chromadb_builtin',
                'available_collections': [col.name for col in self.collections]
            }
        except Exception as e:
            return {
                'collection_name': self.active_collection.name if self.active_collection else 'None',
                'total_chunks': 0,
                'error': str(e)
            }

class ClaudeRAGEngine:
    """Advanced Claude RAG engine with sophisticated prompting"""
    
    def __init__(self):
        self.system_prompt = """You are an expert aviation training assistant with access to comprehensive aviation documentation through a vector database. Your responses are powered by semantic search and retrieval-augmented generation.

Your capabilities:
- Access to aviation documents through semantic vector search
- Contextual understanding of aviation procedures, regulations, and safety
- Personalized training responses based on retrieved document content
- Quiz generation based on specific document sections
- Progress tracking and adaptive learning recommendations

Guidelines:
1. Always ground your responses in the provided context documents
2. Cite specific document sections when relevant
3. Provide practical, actionable aviation knowledge
4. Maintain focus on safety and regulatory compliance
5. Adapt your language to the user's expertise level
6. Encourage questions and deeper exploration of topics"""

    def create_context_aware_prompt(self, query: str, retrieved_docs: List[Dict], session_context: Dict = None) -> str:
        """Create sophisticated RAG prompt with context awareness"""
        
        # Analyze retrieved documents
        doc_analysis = self._analyze_documents(retrieved_docs)
        
        # Format context with relevance ranking
        context_section = ""
        for i, doc_info in enumerate(retrieved_docs[:5], 1):
            similarity_score = doc_info.get('similarity_score', 0)
            metadata = doc_info.get('metadata', {})
            content = doc_info.get('text', doc_info.get('content', ''))
            
            context_section += f"""
Document {i} (Relevance: {similarity_score:.2f}):
Source: {metadata.get('filename', 'Unknown')} - {metadata.get('document_type', 'Unknown type')}
Content: {content[:500]}{'...' if len(content) > 500 else ''}
---"""
        
        # Session context if available
        session_section = ""
        if session_context:
            recent_topics = session_context.get('recent_topics', [])
            if recent_topics:
                session_section = f"\nRecent session topics: {', '.join(recent_topics[-3:])}"
        
        # Construct comprehensive prompt
        prompt = f"""Based on the following aviation documentation retrieved through semantic search, provide a comprehensive response to the user's question.

RETRIEVED DOCUMENTS:
{context_section}

DOCUMENT ANALYSIS:
- Total documents retrieved: {len(retrieved_docs)}
- Document types: {', '.join(doc_analysis['types'])}
- Average relevance score: {doc_analysis['avg_relevance']:.2f}
- Key topics identified: {', '.join(doc_analysis['key_topics'][:5])}

{session_section}

USER QUERY: {query}

RESPONSE REQUIREMENTS:
1. Provide a direct, comprehensive answer based on the retrieved documents
2. Reference specific documents when citing information
3. Include relevant safety considerations or regulatory requirements
4. Suggest follow-up questions or related topics for deeper learning
5. If the retrieved documents don't fully answer the question, clearly state what additional information would be needed

Please structure your response clearly and make it educational for aviation training purposes."""

        return prompt

    def _analyze_documents(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """Analyze retrieved documents for context awareness"""
        if not retrieved_docs:
            return {'types': [], 'avg_relevance': 0, 'key_topics': []}
        
        doc_types = set()
        relevance_scores = []
        all_content = ""
        
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            if 'document_type' in metadata:
                doc_types.add(metadata['document_type'])
            
            score = doc.get('similarity_score', 0)
            if score > 0:
                relevance_scores.append(score)
            
            content = doc.get('text', doc.get('content', ''))
            all_content += " " + content[:200]  # Sample content for topic extraction
        
        # Simple keyword-based topic extraction
        aviation_keywords = ['aircraft', 'flight', 'pilot', 'navigation', 'safety', 'engine', 'weather', 
                           'runway', 'takeoff', 'landing', 'altitude', 'airspeed', 'radio', 'fuel']
        
        key_topics = []
        content_lower = all_content.lower()
        for keyword in aviation_keywords:
            if keyword in content_lower:
                key_topics.append(keyword)
        
        return {
            'types': list(doc_types),
            'avg_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            'key_topics': key_topics[:10]
        }

    def query_claude(self, prompt: str, max_tokens: int = 1500) -> str:
        """Query Claude with sophisticated error handling"""
        client = get_anthropic_client()
        if not client:
            return "❌ Claude not available: No valid API key found. Please configure ANTHROPIC_API_KEY or use the MCP server's API key management system."
        
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=0.3,  # Slightly creative but focused
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        
        except anthropic.APIError as e:
            return f"Claude API Error: {str(e)}. Please check your API key configuration."
        except Exception as e:
            return f"Error generating response: {str(e)}. Please try again with a simpler query."

    def generate_quiz_from_docs(self, topic: str, retrieved_docs: List[Dict], difficulty: str = "intermediate") -> str:
        """Generate contextual quiz questions from retrieved documents"""
        
        if not retrieved_docs:
            return json.dumps([{
                "question": f"What is a key safety consideration in {topic}?",
                "options": ["Follow procedures", "Ignore protocols", "Skip checks", "Operate without documentation"],
                "correct_index": 0,
                "explanation": "Following established procedures is always the primary safety consideration.",
                "source": "General aviation knowledge"
            }])
        
        # Create focused context for quiz generation
        quiz_context = ""
        for i, doc in enumerate(retrieved_docs[:3], 1):
            content = doc.get('text', doc.get('content', ''))
            metadata = doc.get('metadata', {})
            quiz_context += f"Document {i}: {content[:400]}...\n"
        
        quiz_prompt = f"""Based on the following aviation documentation about {topic}, create {difficulty}-level quiz questions for aviation training.

DOCUMENTATION:
{quiz_context}

Create 2-3 multiple choice questions that:
1. Test understanding of key concepts from the provided documentation
2. Focus on practical application rather than memorization
3. Include realistic aviation scenarios
4. Have clear correct answers with educational explanations

Return ONLY a valid JSON array of question objects with this exact format:
[
  {{
    "question": "Question text here",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 0,
    "explanation": "Why this answer is correct",
    "source": "Which document section this came from"
  }}
]"""

        return self.query_claude(quiz_prompt, max_tokens=2000)

# Initialize components
vector_store = ChromaVectorStore()
claude_engine = ClaudeRAGEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    try:
        collection_info = vector_store.get_collection_info()
        return jsonify({
            "status": "healthy",
            "service": "doc-db-advanced-rag",
            "components": {
                "chromadb": "connected",
                "sentence_transformers": "loaded",
                "claude": "initialized",
                "nltk": "available"
            },
            "vector_store": collection_info,
            "active_training_sessions": len(TRAINING_SESSIONS)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Upload endpoint removed - documents already in ChromaDB

@app.route('/query/rag', methods=['POST'])
def advanced_rag_query():
    """Advanced RAG query with Claude integration and context awareness"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        n_results = min(data.get('n_results', 5), 15)
        session_id = data.get('session_id')
        filters = data.get('filters', {})
        
        if not query.strip():
            return jsonify({"error": "Query is required"}), 400
        
        # Perform document search using existing ChromaDB data
        search_results = vector_store.search_documents(
            query=query, 
            n_results=n_results,
            filters=filters if filters else None
        )
        
        # Prepare document info for Claude
        retrieved_docs = []
        for doc, metadata, similarity in zip(
            search_results['documents'],
            search_results['metadatas'],
            search_results['similarity_scores']
        ):
            retrieved_docs.append({
                'text': doc,
                'metadata': metadata,
                'similarity_score': similarity
            })
        
        # Get session context if available
        session_context = None
        if session_id and session_id in TRAINING_SESSIONS:
            session_context = TRAINING_SESSIONS[session_id]
        
        # Generate Claude response
        rag_prompt = claude_engine.create_context_aware_prompt(
            query=query,
            retrieved_docs=retrieved_docs,
            session_context=session_context
        )
        
        claude_response = claude_engine.query_claude(rag_prompt)
        
        # Update session if provided
        if session_id and session_id in TRAINING_SESSIONS:
            session = TRAINING_SESSIONS[session_id]
            session['conversation_history'].append({
                'user_query': query,
                'claude_response': claude_response[:500] + "..." if len(claude_response) > 500 else claude_response,
                'documents_retrieved': len(retrieved_docs),
                'top_similarity_score': max(search_results['similarity_scores']) if search_results['similarity_scores'] else 0,
                'timestamp': datetime.now().isoformat()
            })
            
            # Track topics
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            aviation_keywords = {'aircraft', 'flight', 'pilot', 'safety', 'navigation', 'weather', 'engine'}
            relevant_topics = list(query_words.intersection(aviation_keywords))
            session.setdefault('recent_topics', []).extend(relevant_topics)
            session['recent_topics'] = session['recent_topics'][-10:]  # Keep last 10 topics
        
        return jsonify({
            "query": query,
            "response": claude_response,
            "retrieval_info": {
                "documents_found": len(retrieved_docs),
                "top_similarity_score": max(search_results['similarity_scores']) if search_results['similarity_scores'] else 0,
                "average_similarity": sum(search_results['similarity_scores']) / len(search_results['similarity_scores']) if search_results['similarity_scores'] else 0
            },
            "retrieved_documents": [
                {
                    "content_preview": doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'],
                    "metadata": doc['metadata'],
                    "similarity_score": doc['similarity_score']
                }
                for doc in retrieved_docs[:3]  # Return top 3 for response
            ],
            "session_id": session_id
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing RAG query: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def simple_query():
    """Simplified query endpoint for frontend compatibility"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)
        include_context = data.get('include_context', True)
        
        if not query.strip():
            return jsonify({"error": "Query is required"}), 400
        
        # Perform document search using existing ChromaDB data
        search_results = vector_store.search_documents(
            query=query, 
            n_results=top_k
        )
        
        if not search_results['documents']:
            return jsonify({
                "answer": "I couldn't find specific information about that in the aviation handbook. Could you try rephrasing your question or asking about specific flight operations, aircraft systems, or aviation procedures?",
                "sources": [],
                "confidence": 0,
                "execution_time_ms": 0,
                "tokens_used": 0,
                "cost": 0
            })
        
        # Initialize Claude client
        anthropic_client = initialize_anthropic_client()
        if not anthropic_client:
            return jsonify({"error": "Claude client not available"}), 503
        
        # Prepare context from search results
        context_parts = []
        sources = []
        for i, (doc, metadata) in enumerate(zip(search_results['documents'], search_results['metadatas'])):
            context_parts.append(f"Document {i+1}: {doc[:500]}...")
            sources.append(f"Airplane Flying Handbook - Section {i+1}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for Claude
        prompt = f"""You are an expert aviation instructor with deep knowledge of the FAA Airplane Flying Handbook. Answer the user's question based on the provided documentation context.

Context from Airplane Flying Handbook:
{context}

User Question: {query}

Please provide a comprehensive, accurate answer based on the aviation handbook content. Focus on practical flying knowledge, safety procedures, and official FAA guidance. If the context doesn't contain enough information to answer fully, say so and suggest what type of aviation topics you can help with.

Keep your response informative but conversational, as if you're teaching a student pilot."""

        start_time = datetime.now()
        
        # Get response from Claude
        try:
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            claude_response = response.content[0].text
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            return jsonify({
                "answer": claude_response,
                "sources": sources[:3],  # Return top 3 sources
                "confidence": max(search_results['similarity_scores']) if search_results['similarity_scores'] else 0,
                "execution_time_ms": execution_time,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else 0,
                "cost": 0.001  # Approximate cost
            })
            
        except Exception as claude_error:
            print(f"Claude API error: {claude_error}")
            return jsonify({
                "answer": "I encountered an issue processing your question with the AI system. Please try rephrasing your question or asking about basic aviation concepts like the four forces of flight, aircraft systems, or flight procedures.",
                "sources": sources[:3],
                "confidence": 0,
                "execution_time_ms": 0,
                "tokens_used": 0,
                "cost": 0,
                "error": "AI processing unavailable"
            })
        
    except Exception as e:
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the RAG system"""
    try:
        collection_info = vector_store.get_collection_info()
        return jsonify({
            "total_documents": 1,  # We have 1 PDF document
            "chunks_count": collection_info.get("total_chunks", 399),
            "embedding_model": "chromadb-builtin",
            "last_updated": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "total_documents": 1,
            "chunks_count": 399,
            "embedding_model": "chromadb-builtin", 
            "last_updated": datetime.now().isoformat()
        })

@app.route('/quiz/generate', methods=['POST'])
def generate_contextual_quiz():
    """Generate quiz questions based on retrieved documents"""
    try:
        data = request.get_json()
        topic = data.get('topic', 'aviation')
        difficulty = data.get('difficulty', 'intermediate')
        num_questions = min(data.get('num_questions', 3), 5)
        
        # Search for relevant documents
        search_results = vector_store.search_documents(topic, n_results=5)
        
        if not search_results['documents']:
            return jsonify({
                "error": "No relevant documents found for quiz generation",
                "suggestion": "Please upload documents related to this topic first"
            }), 404
        
        # Prepare retrieved docs for quiz generation
        retrieved_docs = []
        for doc, metadata, similarity in zip(
            search_results['documents'],
            search_results['metadatas'], 
            search_results['similarity_scores']
        ):
            retrieved_docs.append({
                'text': doc,
                'metadata': metadata,
                'similarity_score': similarity
            })
        
        # Generate quiz using Claude
        quiz_json = claude_engine.generate_quiz_from_docs(topic, retrieved_docs, difficulty)
        
        try:
            questions = json.loads(quiz_json)
            if not isinstance(questions, list):
                questions = [questions]
        except json.JSONDecodeError:
            # Fallback questions if Claude response isn't valid JSON
            questions = [{
                "question": f"Based on the documentation about {topic}, what is the most important consideration?",
                "options": ["Following established procedures", "Operating without documentation", "Ignoring safety protocols", "Skipping required checks"],
                "correct_index": 0,
                "explanation": "Following established procedures is always the most important consideration in aviation.",
                "source": "Retrieved aviation documentation"
            }]
        
        quiz = {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "difficulty": difficulty,
            "questions": questions[:num_questions],
            "created_at": datetime.now().isoformat(),
            "based_on_documents": len(retrieved_docs),
            "source_documents": [
                {
                    "filename": doc['metadata'].get('filename', 'Unknown'),
                    "document_type": doc['metadata'].get('document_type', 'Unknown'),
                    "similarity_score": doc['similarity_score']
                }
                for doc in retrieved_docs[:3]
            ]
        }
        
        return jsonify(quiz)
        
    except Exception as e:
        return jsonify({"error": f"Error generating quiz: {str(e)}"}), 500

@app.route('/training/start', methods=['POST'])
def start_training_session():
    """Start advanced training session with vector-powered context"""
    try:
        data = request.get_json() or {}
        topic = data.get('topic', 'aviation training')
        learning_objectives = data.get('learning_objectives', [])
        
        session_id = str(uuid.uuid4())
        
        # Search for initial relevant content
        search_results = vector_store.search_documents(topic, n_results=5)
        
        session = {
            "id": session_id,
            "topic": topic,
            "learning_objectives": learning_objectives,
            "created_at": datetime.now().isoformat(),
            "conversation_history": [],
            "recent_topics": [],
            "progress": {
                "queries_made": 0,
                "documents_accessed": len(search_results['documents']),
                "average_relevance": sum(search_results['similarity_scores']) / len(search_results['similarity_scores']) if search_results['similarity_scores'] else 0
            },
            "vector_store_context": {
                "total_documents_available": vector_store.get_collection_info()['total_chunks'],
                "initial_topic_relevance": max(search_results['similarity_scores']) if search_results['similarity_scores'] else 0
            }
        }
        
        TRAINING_SESSIONS[session_id] = session
        
        # Generate welcome message using Claude if documents are available
        welcome_message = "Welcome to your advanced aviation training session!"
        
        if search_results['documents']:
            # Prepare context for welcome message
            context_preview = search_results['documents'][0][:200] + "..."
            welcome_prompt = f"""Create a welcoming, encouraging introduction for an aviation training session about '{topic}'.

Available documentation includes content such as: "{context_preview}"

The introduction should:
1. Welcome the student warmly
2. Mention that you have access to comprehensive aviation documentation
3. Encourage them to ask specific questions about {topic}
4. Be concise but motivating (2-3 sentences)

Do not mention technical details about vector databases or RAG systems."""
            
            welcome_message = claude_engine.query_claude(welcome_prompt, max_tokens=300)
        
        return jsonify({
            "session_id": session_id,
            "topic": topic,
            "welcome_message": welcome_message,
            "session_capabilities": {
                "semantic_document_search": True,
                "claude_powered_responses": True,
                "contextual_quiz_generation": True,
                "progress_tracking": True
            },
            "available_content": {
                "total_document_chunks": session["vector_store_context"]["total_documents_available"],
                "topic_relevant_documents": len(search_results['documents']),
                "initial_relevance_score": session["vector_store_context"]["initial_topic_relevance"]
            },
            "status": "active"
        })
        
    except Exception as e:
        return jsonify({"error": f"Error starting training session: {str(e)}"}), 500

@app.route('/vector-store/info', methods=['GET'])
def vector_store_info():
    """Get comprehensive vector store information"""
    try:
        return jsonify(vector_store.get_collection_info())
    except Exception as e:
        return jsonify({"error": f"Error getting vector store info: {str(e)}"}), 500

@app.route('/collections/list', methods=['GET'])
def list_collections():
    """List all available ChromaDB collections"""
    try:
        collections = chroma_client.list_collections()
        collection_info = []
        
        for col in collections:
            try:
                count = col.count()
                collection_info.append({
                    'name': col.name,
                    'document_count': count
                })
            except Exception as e:
                collection_info.append({
                    'name': col.name,
                    'document_count': 0,
                    'error': str(e)
                })
        
        return jsonify({
            "collections": collection_info,
            "total_collections": len(collection_info)
        })
    except Exception as e:
        return jsonify({"error": f"Error listing collections: {str(e)}"}), 500

# Admin endpoints for PDF processing
@app.route('/admin/pdf/list', methods=['GET'])
def admin_list_pdfs():
    """List all PDF files in the pdfs folder"""
    try:
        from robust_pdf_processor import RobustPDFProcessor
        processor = RobustPDFProcessor(chroma_client)
        pdf_files = processor.get_pdf_files()
        
        return jsonify({
            "success": True,
            "pdf_files": [
                {
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                } for f in pdf_files
            ],
            "total_files": len(pdf_files)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error listing PDF files: {str(e)}"
        }), 500

@app.route('/admin/pdf/process', methods=['POST'])
def admin_process_pdfs():
    """Process all PDF files in the pdfs folder"""
    try:
        from robust_pdf_processor import RobustPDFProcessor
        processor = RobustPDFProcessor(chroma_client)
        
        # Get reset parameter from request
        reset_collection = request.json.get('reset_collection', True) if request.is_json else True
        
        result = processor.process_all_pdfs(reset_collection=reset_collection)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error processing PDFs: {str(e)}"
        }), 500

@app.route('/admin/pdf/status', methods=['GET'])
def admin_processing_status():
    """Get current PDF processing status"""
    try:
        from robust_pdf_processor import RobustPDFProcessor
        processor = RobustPDFProcessor(chroma_client)
        
        status = processor.get_processing_status()
        stats = processor.get_collection_stats()
        
        return jsonify({
            "success": True,
            "processing_status": status,
            "collection_stats": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error getting status: {str(e)}"
        }), 500

@app.route('/admin/collection/stats', methods=['GET'])
def admin_collection_stats():
    """Get detailed collection statistics"""
    try:
        print(f"DEBUG: CHROMADB_PATH = {CHROMADB_PATH}")
        from robust_pdf_processor import RobustPDFProcessor
        processor = RobustPDFProcessor(chroma_client)
        
        stats = processor.get_collection_stats()
        print(f"DEBUG: stats = {stats}")
        
        return jsonify({
            "success": True,
            "stats": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error getting collection stats: {str(e)}"
        }), 500

@app.route('/admin/collection/reset', methods=['POST'])
def admin_reset_collection():
    """Reset/clear the ChromaDB collection"""
    try:
        collection_name = "rag_documents"
        
        # Delete existing collection
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        except:
            pass  # Collection might not exist
        
        # Create new empty collection
        embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"description": "RAG document chunks with embeddings"}
        )
        
        return jsonify({
            "success": True,
            "message": f"Collection '{collection_name}' has been reset",
            "collection_name": collection_name
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error resetting collection: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Query-Only RAG Service for Doc-DB...")
    print("📚 Using existing ChromaDB collections (PDF already uploaded)")
    print("🔍 ChromaDB built-in text search (no ML dependencies)")
    print("🤖 Claude-powered intelligent responses")
    print("🎯 Context-aware training and quiz generation")
    print(f"📍 Using ChromaDB at {CHROMADB_PATH}")
    print("🌐 API available on http://localhost:8005")
    print("")
    
    # Initialize Anthropic client using MCP server's approach
    print("🔐 Initializing Anthropic Claude client...")
    client = initialize_anthropic_client()
    if client:
        print("✅ Claude integration ready!")
    else:
        print("⚠️  Claude integration not available - basic text search only")
        print("   To enable: Set ANTHROPIC_API_KEY environment variable or use MCP API key management")
    
    print("")
    # Configure Flask app
    PORT = int(os.getenv('PORT', 8005))
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)