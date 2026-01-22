# Personalized Product Recommendation Narratives with a 60M-parameter Small Language Model

**From-scratch SLM + LoRA + RAG for engaging, low-cost, privacy-friendly e-commerce recommendations**

![Project Banner / Demo GIF Placeholder]
*(Add a short GIF or screenshot here later – e.g., Streamlit UI showing input → generated narrative)*

## 🎯 Project Goal

Build an **end-to-end, portfolio-grade ML project** that demonstrates:

- Training a Small Language Model (~50–60M parameters) from scratch (inspired by nanoGPT)
- Efficient fine-tuning with LoRA and quantization
- Retrieval-Augmented Generation (RAG) for factual, up-to-date recommendations
- Personalized narrative generation for e-commerce (beyond simple "you might like X")
- Full ML engineering spectrum: data → modeling → evaluation → production → MLOps

The model generates **coherent, engaging recommendation text** like:

> "You've loved our bold single-origin coffees in the past — especially that bright Ethiopian roast. For your next morning brew, try this new Guatemalan Finca El Platanillo: deep chocolate notes with a hint of red berry, perfectly balanced for pour-over. Currently 20% off — pairs beautifully with your favorite oat milk!"

...while grounding recommendations in real product data via RAG, all running efficiently on modest hardware.

## Why This Matters (2025–2026 Context)

- Massive LLMs (70B+) are expensive and slow for real-time personalization at scale
- Small models + PEFT + RAG offer **sub-second inference**, **low carbon footprint**, and **on-device potential**
- Narrative recommendations outperform list-based ones in user engagement & conversion (industry studies 2024–2025)
- Privacy: keep user history local, only send minimal query to server

## Key Features / Techniques Demonstrated

- Decoder-only Transformer built from scratch (causal self-attention, rotary? no — learned positional)
- Efficient training on memmapped binaries (Tiktoken + GPT-2 vocab)
- Parameter-efficient fine-tuning with LoRA (rank 16–32)
- 8-bit / 4-bit quantization (bitsandbytes)
- FAISS-based RAG retriever over product embeddings
- Controlled generation (temperature, top-k, repetition penalty)
- Experiment tracking with MLflow
- FastAPI backend + Streamlit interactive demo
- Docker support for easy deployment
- Bias & fairness checks on recommendations

## Repository Structure

personalized-recommendation-narratives-slm/
│
├── README.md                     ← main entry point, project overview, demo GIF/video link, setup instructions
│
├── requirements.txt              ← pinned dependencies
├── setup.py                      ← optional, if we want to make it installable
│
├── notebooks/
│   ├── 01_project_overview_and_motivation.ipynb
│   ├── 02_data_exploration_and_preparation.ipynb
│   ├── 03_slm_architecture_from_scratch.ipynb          ← heavily based on your original notebook + explanations
│   ├── 04_adding_lora_and_quantization.ipynb
│   ├── 05_building_rag_retriever.ipynb
│   ├── 06_fine_tuning_and_experiment_tracking.ipynb
│   ├── 07_evaluation_and_quality_assessment.ipynb
│   └── 08_inference_examples_and_controlled_generation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               ← custom Dataset + collate_fn
│   │   ├── preprocessing.py         ← tokenization, prompt templates, memmap helpers
│   │   └── rag_retriever.py         ← FAISS index builder + search logic
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── gpt.py                   ← core GPT architecture (from your notebook + modifications)
│   │   ├── lora.py                  ← LoRA wrappers / injection logic
│   │   └── utils.py                 ← init weights, generation helpers, KV cache
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               ← training loop, logging, checkpointing
│   │   └── evaluate.py              ← metrics computation
│   │
│   └── inference/
│       ├── __init__.py
│       └── generator.py             ← high-level recommendation generation with RAG
│
├── app/
│   ├── main.py                      ← FastAPI backend
│   ├── streamlit_app.py             ← frontend demo
│   └── utils.py                     ← API helpers
│
├── configs/
│   ├── base.yaml                    ← hydra-style config (model, training, data)
│   └── inference.yaml
│
├── scripts/
│   ├── download_data.py             ← helper to download + subsample HF datasets
│   ├── prepare_data.py              ← creates .bin files
│   ├── train.py                     ← entry point for training
│   └── generate_demo.py             ← example generations
│
├── mlruns/                          ← MLflow tracking folder (gitignored)
├── .gitignore
├── docker/
│   └── Dockerfile
└── docs/
    └── architecture.md              ← mermaid diagrams, explanations