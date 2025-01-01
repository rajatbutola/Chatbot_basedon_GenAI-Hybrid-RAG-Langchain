# Chatbot_basedon_GenAI-Hybrid-RAG-Langchain

# Chatbot with RAG and Hybrid Retrieval  

This repository provides an advanced system for generating chatbot using **Retrieval-Augmented Generation (RAG)** techniques and an ensemble of retrievers for improved query understanding. It incorporates few-shot learning to fine-tune responses and supports history-aware queries to enhance context handling in conversational settings.  

## Key Features  

- **FAISS Local Index Retrieval**: Efficiently loads and queries a FAISS-based vector store for similarity search.  
- **BM25 and Ensemble Retrieval**: Combines vector-based and keyword-based retrieval methods (FAISS and BM25) with weighted contributions to enhance the accuracy of retrieved information.  
- **Few-Shot Learning**: Uses example-driven prompts to improve the generation of ChimeraX commands for visualization, analysis, and data manipulation tasks.  
- **History-Aware Question Reformulation**: Reformulates user queries into standalone questions by leveraging chat history, ensuring contextually relevant responses.  
- **Customizable Prompts**: Supports tailored prompt templates for specific tasks like negotiation, command generation, and question contextualization.  
- **Conversation Memory Management**: Implements memory to maintain context across multiple interactions using `ConversationBufferMemory`.  
- **RAG Chain Integration**: Combines retrieval and language modeling using a multi-step RAG pipeline for accurate and context-sensitive responses.  

## Components  

1. **Retrievers**:  
   - FAISS-based similarity search retriever.  
   - BM25 keyword retriever.  
   - Weighted ensemble retriever combining FAISS and BM25.  

2. **LLM**:  
   - GPT-4 Turbo is used as the language model for command generation and query reformulation.  

3. **Prompts**:  
   - Few-shot prompt examples for ChimeraX commands.  
   - Custom negotiation prompts to tailor responses for actionable tasks.  
   - Contextual question formulation prompts for history-aware interactions.  

4. **Memory**:  
   - Maintains and returns chat history using a conversation buffer for better context management.  

## Use Cases  

- **Text Generation**: Generate precise text to handle tasks like protein visualization, analysis, and ligand manipulation.  
- **Scientific Query Resolution**: Supports researchers by enabling contextual understanding and command formulation for molecular visualization.  
- **Conversational Context Handling**: Reformulates and clarifies user queries with historical context for effective responses.  

## Installation  

To use this repository, ensure you have the following dependencies installed:  

- Python 3.8+  
- `transformers`  
- `datasets`  
- `faiss-cpu` or `faiss-gpu`  
- `tqdm`  
- `openai`  
- `langchain`  

Install the required packages using:  
```bash
pip install transformers datasets faiss-cpu tqdm openai langchain
```

## Usage
- Load the FAISS Index
- Ensure the FAISS index is available locally. Update the path in the script if necessary.

## Run the Script
Execute the main script to query the system:

```bash
python main.py
```
## File Structure
- main.py: Main script for setting up retrievers, LLM, and RAG chain.
- faiss_local_db/: Directory containing the FAISS index.
- examples/: Few-shot learning examples for ChimeraX command generation


