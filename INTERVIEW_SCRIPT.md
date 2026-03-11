# Explaining Your Project in an Interview: NotebookLM Clone

## Intro - Setting the Stage
"So, I'd love to tell you about my project, NotebookLM Clone. I built it because I wanted to create an open-source version of Google's NotebookLM, which helps users ground AI responses in their own documents with accurate citations. My motivation was to explore document-grounded AI, improve information retrieval, and make advanced AI tools accessible for research and learning."

## Technologies - The Tools of the Trade
"For this project, I used Python as the main language, Streamlit for the web UI, ChromaDB for vector database storage, PyMuPDF for document parsing, AssemblyAI for audio transcription, Firecrawl for web scraping, and CrewAI for agent orchestration. I chose Python because of its rich ecosystem for AI and data processing. Streamlit was ideal for rapid UI development, and ChromaDB provided efficient semantic search. Each technology was selected for its suitability and performance in handling multi-format data and AI workflows."

## Action - The Journey of Development
"The project flow involved several steps:
- Document ingestion (PDF, text, audio, YouTube, web)
- Content extraction and chunking
- Embedding generation
- Storing vectors in ChromaDB
- Semantic search and retrieval
- Generating cited answers and podcasts
For example, when implementing the vector database, I initially tried Milvus but switched to ChromaDB due to compatibility and ease of use. A challenge was optimizing pipeline initialization, which I solved by lazy-loading optional components and adding progress indicators. I learned the importance of modular design and version management for complex dependencies."

## Result - The Final Product
"The final result is a functional web application that allows users to upload documents, ask questions, and receive cited answers. It also generates AI podcasts from documents. You can see it live locally, and the code is available on GitHub. I'm proud of the intuitive UI, multi-format support, and the accuracy of AI-generated citations."

---

## Project Workflow

1. User uploads documents (PDF, text, audio, YouTube, web).
2. Content is extracted and chunked for context preservation.
3. Chunks are embedded into vectors using FastEmbed.
4. Vectors are stored in ChromaDB with citation metadata.
5. User asks questions; queries are embedded and searched in ChromaDB.
6. Relevant chunks are retrieved and cited in AI responses.
7. Optional: Podcast scripts and TTS audio are generated from documents.

---

## 10 Probable Interview Questions (with Answers)

1. **What motivated you to build an open-source NotebookLM Clone?**
	- I wanted to make document-grounded AI accessible and transparent, inspired by Google's NotebookLM. My motivation was to explore retrieval-augmented generation and help users get cited, trustworthy answers from their own documents.

2. **Why did you choose Python and Streamlit for this project?**
	- Python offers a rich ecosystem for AI and data processing. Streamlit is ideal for rapid UI development and easy integration with Python backends, making it perfect for prototyping and deploying AI apps.

3. **How does ChromaDB work as a vector database, and why did you select it over Milvus?**
	- ChromaDB stores document embeddings and enables fast vector similarity search. I chose it over Milvus due to easier setup, compatibility, and its Python-native API, which fit my workflow better.

4. **Can you explain the document ingestion and chunking process?**
	- Documents are parsed (PDF, text, audio, web), then split into overlapping chunks to preserve context. Each chunk is embedded and stored with metadata for citation and retrieval.

5. **How do you ensure accurate citations in AI responses?**
	- Each answer includes references to the original document chunks, with page numbers or timestamps. The retrieval pipeline is designed to always return source metadata alongside generated responses.

6. **What challenges did you face with dependency management and how did you resolve them?**
	- I encountered version conflicts and missing packages, especially with vector DBs and agent frameworks. I resolved these by pinning compatible versions, switching to ChromaDB, and deferring optional component initialization.

7. **How does the project handle multi-format data (PDF, audio, YouTube, web)?**
	- Specialized processors handle each format: PyMuPDF for documents, AssemblyAI for audio, Firecrawl for web, and YouTube transcriber for videos. All are unified in the pipeline for seamless ingestion and retrieval.

8. **What is the role of CrewAI in your workflow?**
	- CrewAI orchestrates agent-based tasks, such as document processing and response generation, allowing modular and scalable AI workflows.

9. **How did you optimize the pipeline initialization for performance?**
	- I implemented lazy loading for optional components and added progress indicators, which reduced startup time and improved user experience.

10. **If you were to extend this project, what features would you add next?**
	- I would add real-time collaboration, more advanced memory layers, and support for additional document formats. Integrating more LLMs and improving citation granularity are also on my roadmap.
