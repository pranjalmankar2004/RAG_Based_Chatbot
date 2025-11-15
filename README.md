Legal RAG Chatbot

A Retrieval-Augmented Generation (RAG) based legal assistant that answers Indian law–related questions using information extracted from legal PDFs. The system combines PDF retrieval with large language models (LLMs) to produce clear, human-readable responses backed by citations.

Features

Retrieval-Augmented Generation using embeddings and semantic search

Human-friendly answers generated using Groq Llama and Google AI models

Five+ legal documents pre-indexed (Acts, Rules, Guidelines, Case Laws)

Accurate source citations for each response

Professional HTML, CSS, JS frontend

FastAPI backend

Clean, minimal project structure

Documents Included

The system indexes PDFs stored in the documents/ folder. These include:

Right to Information Act, 2005

Consumer Protection Act

Indian Evidence Act

IT Intermediary Rules

Supreme Court Landmark Judgments

Additional regulations and guidelines relevant to the legal domain

How It Works

PDF documents are parsed and cleaned.

Text is split into structured semantic chunks.

Embeddings are generated for retrieval.

When a user asks a question, the system retrieves the top-ranked chunks.

The LLM rewrites the information into clear human language.

The final answer is delivered with citations and confidence score.
