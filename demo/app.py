"""
HuggingFace Spaces Demo for Industrial Audio RAG
A streamlined demo interface showcasing the RAG system capabilities
"""

import gradio as gr
import requests
import json
import os
from typing import Dict, List, Optional
import time

# Demo configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "https://industrial-audio-rag-demo.herokuapp.com")
API_KEY = os.environ.get("API_KEY", "")

# Sample queries for demo
SAMPLE_QUERIES = [
    "Which bearing clips in section 00 show dominant frequency above 900 Hz?",
    "Find all clips with bearing anomalies in the training dataset",
    "What are the frequency characteristics of normal vs abnormal bearings?",
    "Show me clips with high energy content in the 1-2 kHz range",
    "Which audio files contain bearing defects similar to outer race damage?",
    "Find clips with spectral patterns indicating bearing wear",
    "What's the difference between normal and abnormal bearing signatures?",
    "Show bearing clips with unusual vibration patterns"
]

def make_api_request(query: str, max_results: int = 5) -> Dict:
    """Make request to the RAG API"""
    try:
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        
        payload = {
            "query": query,
            "max_results": max_results
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API Error {response.status_code}: {response.text}",
                "answer": "Sorry, I encountered an error processing your request.",
                "sources": []
            }
    
    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout",
            "answer": "The request took too long to process. Please try again.",
            "sources": []
        }
    except Exception as e:
        return {
            "error": str(e),
            "answer": "An unexpected error occurred. Please try again later.",
            "sources": []
        }

def format_sources(sources: List[Dict]) -> str:
    """Format source information for display"""
    if not sources:
        return "No sources found."
    
    formatted = "### 📁 Sources Found:\n\n"
    for i, source in enumerate(sources[:5], 1):
        filename = source.get('filename', 'Unknown file')
        score = source.get('score', 0)
        snippet = source.get('snippet', 'No snippet available')
        
        formatted += f"**{i}. {filename}** (Relevance: {score:.3f})\n"
        formatted += f"```\n{snippet[:200]}{'...' if len(snippet) > 200 else ''}\n```\n\n"
    
    return formatted

def format_metrics(response_data: Dict) -> str:
    """Format performance metrics"""
    if 'metadata' not in response_data:
        return ""
    
    metadata = response_data['metadata']
    metrics = []
    
    if 'response_time' in metadata:
        metrics.append(f"⏱️ Response Time: {metadata['response_time']:.2f}s")
    
    if 'embedding_time' in metadata:
        metrics.append(f"🔍 Embedding Time: {metadata['embedding_time']:.3f}s")
    
    if 'search_time' in metadata:
        metrics.append(f"🔎 Search Time: {metadata['search_time']:.3f}s")
    
    if 'llm_time' in metadata:
        metrics.append(f"🤖 LLM Time: {metadata['llm_time']:.2f}s")
    
    if 'total_sources' in metadata:
        metrics.append(f"📊 Sources Found: {metadata['total_sources']}")
    
    if metrics:
        return "### 📈 Performance Metrics\n" + " | ".join(metrics) + "\n\n"
    
    return ""

def query_rag_system(query: str, max_results: int, progress=gr.Progress()) -> tuple:
    """Main function to query the RAG system"""
    if not query.strip():
        return "Please enter a question about industrial audio data.", "", ""
    
    progress(0.1, desc="Sending query...")
    
    start_time = time.time()
    response_data = make_api_request(query, max_results)
    end_time = time.time()
    
    progress(0.9, desc="Formatting response...")
    
    # Extract response components
    answer = response_data.get('answer', 'No answer provided.')
    sources = response_data.get('sources', [])
    error = response_data.get('error')
    
    # Format the response
    if error:
        formatted_answer = f"❌ **Error:** {error}\n\n{answer}"
    else:
        formatted_answer = f"💡 **Answer:**\n\n{answer}"
    
    # Add performance metrics
    if 'metadata' not in response_data:
        response_data['metadata'] = {'response_time': end_time - start_time}
    
    metrics = format_metrics(response_data)
    sources_formatted = format_sources(sources)
    
    progress(1.0, desc="Complete!")
    
    return formatted_answer, metrics, sources_formatted

def get_system_info() -> Dict:
    """Get system information from the API"""
    try:
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        
        response = requests.get(f"{API_BASE_URL}/info", headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    return {
        "status": "Demo Mode",
        "model_info": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "Ollama (Local)"
        },
        "data_info": {
            "total_documents": "~2000 audio clips",
            "collection_name": "DCASE 2024 Task-2"
        }
    }

# Create the Gradio interface
def create_demo():
    """Create the Gradio demo interface"""
    
    system_info = get_system_info()
    
    with gr.Blocks(
        title="Industrial Audio RAG System",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .header { text-align: center; margin-bottom: 2rem; }
        .metrics { background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🏭 Industrial Audio RAG System
        
        **Retrieval-Augmented Generation for Industrial Audio Analysis**
        
        This demo showcases an AI system that can answer questions about industrial audio data, 
        specifically bearing condition monitoring from the DCASE 2024 Task-2 dataset.
        
        🎯 **Ask questions about:**
        - Bearing condition analysis
        - Frequency domain characteristics  
        - Anomaly detection patterns
        - Audio clip comparisons
        """, elem_classes=["header"])
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Ask a question about industrial audio data:",
                    placeholder="e.g., Which bearing clips show dominant frequency above 900 Hz?",
                    lines=2
                )
                
                with gr.Row():
                    max_results = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="Maximum results to retrieve"
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                gr.Markdown("### 💡 Try these sample questions:")
                sample_buttons = []
                for sample in SAMPLE_QUERIES[:4]:
                    btn = gr.Button(sample, variant="secondary", size="sm")
                    sample_buttons.append(btn)
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ System Information")
                system_status = gr.JSON(
                    value=system_info,
                    label="System Status"
                )
        
        # Output sections
        with gr.Row():
            with gr.Column():
                answer_output = gr.Markdown(
                    label="Answer",
                    value="👋 Welcome! Ask a question about industrial audio data to get started."
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                metrics_output = gr.Markdown(
                    label="Performance Metrics",
                    elem_classes=["metrics"]
                )
            
            with gr.Column(scale=2):
                sources_output = gr.Markdown(
                    label="Source Documents"
                )
        
        # Footer information
        gr.Markdown("""
        ---
        ### 📚 About this Demo
        
        This system uses **Retrieval-Augmented Generation (RAG)** to answer questions about industrial audio data:
        
        1. **🔍 Embedding**: Your question is converted to a vector representation
        2. **🔎 Search**: Similar audio clips are found using vector similarity
        3. **🤖 Generation**: An LLM generates answers based on the retrieved context
        
        **Tech Stack:** FastAPI, Qdrant Vector DB, Sentence Transformers, Ollama
        
        **Dataset:** DCASE 2024 Task-2 - Unsupervised Anomalous Sound Detection for Machine Condition Monitoring
        
        🚀 **[View Source Code](https://github.com/otosense/industrial-audio-rag)** | 
        📊 **[Performance Metrics](https://github.com/otosense/industrial-audio-rag/tree/main/eval)** |
        ☁️ **[Deploy Guide](https://github.com/otosense/industrial-audio-rag/tree/main/infra)**
        """)
        
        # Event handlers
        submit_btn.click(
            fn=query_rag_system,
            inputs=[query_input, max_results],
            outputs=[answer_output, metrics_output, sources_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "Ready to answer your questions!", "", ""),
            outputs=[query_input, answer_output, metrics_output, sources_output]
        )
        
        # Sample button handlers
        for i, btn in enumerate(sample_buttons):
            btn.click(
                fn=lambda sample=SAMPLE_QUERIES[i]: sample,
                outputs=query_input
            )
        
        # Auto-refresh system info every 30 seconds
        demo.load(
            fn=get_system_info,
            outputs=system_status,
            every=30
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )