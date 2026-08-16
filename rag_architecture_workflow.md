# 3GPP RAG Chatbot Architecture Workflow

This document outlines the architectural approach for building a Retrieval-Augmented Generation (RAG) chatbot focused on Telecom 3GPP standards, with a critical requirement for near-zero hallucinations. 

We approach this AI architecture problem through a structured workflow: Data Strategy, Retrieval Design, Generation, and Output Verification (Guardrails).

```mermaid
graph TD
    subgraph Data["1. Data Strategy & Ingestion"]
        A1["3GPP Specs PDF/Word"] --> A2["Document Parser"]
        A2 --> A3["Structure-Aware Chunking"]
        A3 --> A4["Metadata Extraction<br/>Spec ID, Release, Section"]
        A4 --> A5["Embedding Model<br/>Domain Specific"]
        A5 --> A6[("Vector DB + Keyword DB")]
    end

    subgraph Client["Client Interface"]
        U1["User Query"]
        U2["Final Response & Citations"]
    end

    subgraph Retrieval["2. Orchestration & Retrieval"]
        R1["Query Rewriting/Expansion"]
        R2["Hybrid Search<br/>Dense + BM25"]
        R3["Re-ranking Model<br/>Cross-Encoder"]
        U1 --> R1
        R1 --> R2
        R2 <--> A6
        R2 --> R3
    end

    subgraph Generation["3. Grounded Generation"]
        G1["Strict System Prompting<br/>'Only use provided context'"]
        G2["LLM Generator"]
        R3 --> G1
        G1 --> G2
    end

    subgraph Guardrails["4. Verification & Guardrails"]
        V1{"Context Entailment Check<br/>Is response supported?"}
        V2["Fallback: Information not found"]
        V3["Citation Formatting"]
        G2 --> V1
        V1 -- No --> V2
        V1 -- Yes --> V3
    end

    V3 --> U2
    V2 --> U2
```

## Architectural Breakdown for Near-Zero Hallucination

### 1. Data Strategy & Ingestion
To minimize hallucinations, the data must be highly structured. 
* **Structure-Aware Chunking:** 3GPP documents are highly technical, structured, and numbered. Chunks must respect section boundaries rather than arbitrary character limits to preserve context.
* **Metadata Extraction:** Tagging chunks with Release (e.g., Rel-17), Spec Series (e.g., 38.331), and section numbers. This allows for precise filtering before semantic search.

### 2. Orchestration & Retrieval
Poor retrieval is the leading cause of RAG hallucinations. If the LLM doesn't get the right facts, it guesses.
* **Hybrid Search:** Combining keyword search (BM25) with semantic vector search. Telecom terms often involve specific acronyms (e.g., "RRC", "NAS", "UPF") where exact keyword matching outperforms semantic matching.
* **Re-ranking:** A Cross-Encoder re-evaluates the retrieved chunks against the query to ensure only the most relevant, highly-correlated chunks are passed to the LLM.

### 3. Grounded Generation
* **Strict Prompting:** The LLM must be explicitly instructed: *"You are a strict 3GPP standards expert. Answer ONLY using the provided context. If the context does not contain the answer, reply exactly with 'The provided 3GPP specifications do not contain information to answer this query.' Do not use outside knowledge."*

### 4. Verification & Guardrails (The Hallucination Filter)
* **Context Entailment Check (Self-Reflection / NLI):** Before returning the response, a secondary check (using a smaller, fast NLI model or the LLM itself) verifies if every claim in the generated response is strictly entailed by the retrieved chunks.
* **Mandatory Citations:** Forcing the model to quote the specific 3GPP spec and section number for every claim. If it cannot cite it, the claim is removed.
