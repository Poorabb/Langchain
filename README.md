# LangChain — Learning Repository

A structured walkthrough of LangChain fundamentals, built while learning to design and ship RAG-based GenAI applications. Each folder builds on the previous one, moving from basic model/prompt handling to full retrieval-augmented generation pipelines.

---

## 📂 Structure

```
├── 1_LANGCHAIN_MODELS/          # LLM & chat model wrappers, model configs
├── 2_LANGCHAIN_PROMPTS/         # Prompt templates, few-shot prompting
├── 3_LANGCHAIN_STRUCTURED_OUTPUT/  # Pydantic schemas, structured/typed outputs
├── 4_Chains/                    # Sequential, parallel & conditional chains
├── 5_Runnables/                 # RunnableParallel, RunnablePassthrough, RunnableLambda
├── 6_RAG/                       # Document loaders, splitters, vector stores, retrievers
├── 7_Tools/                     # Custom tools, StructuredTool, tool-calling agents
├── requirements.txt
└── .env (not tracked)
```

---

## 🧠 What's Covered

- **Models** — working with chat models and LLM wrappers across providers
- **Prompts** — `PromptTemplate`, few-shot examples, dynamic prompt construction
- **Structured Output** — enforcing typed responses using Pydantic models
- **Chains** — composing multi-step LLM workflows
- **Runnables (LCEL)** — the LangChain Expression Language: parallel execution, passthroughs, custom lambdas
- **RAG** — chunking strategies, embeddings (Gemini `gemini-embedding-001`), Pinecone vector store integration, retrieval pipelines
- **Tools** — building custom tools with `StructuredTool`, laying groundwork for agentic workflows

---

## ⚙️ Setup

```bash
python -m venv langenv
langenv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create a `.env` file in the root:

```
GOOGLE_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
HF_TOKEN=your_hugging_face_key_here
```

---

## 🛠️ Stack

LangChain · LangChain Core · Google Gemini · Pinecone · Python

---

## 📌 Notes

This repo is a learning log, not a production codebase — notebooks are exploratory and iterate as concepts are tested. Production-grade implementations (e.g. the YouTube RAG Chatbot extension) live in separate repos built on top of these fundamentals.
