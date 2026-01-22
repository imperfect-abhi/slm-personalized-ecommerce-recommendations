# Personalized Product Recommendation Narratives with a 60M-parameter Small Language Model

**From-scratch SLM + LoRA + RAG for engaging, low-cost, privacy-friendly e-commerce recommendations**



## Project Goal

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


## Quick Start (Local / Colab)

1. Clone the repo
   ```bash
   git clone https://github.com/yourusername/personalized-recommendation-narratives-slm.git
   cd personalized-recommendation-narratives-slm

2. Install dependencies
```pip install -r requirements.txt```

3. (Optional) Download & prepare data subset
```python scripts/download_data.py --subset electronics --max_samples 500000```
```python scripts/prepare_data.py```

4. Explore the notebooks in order (start with notebooks/01_...)

5. Run the demo app
```streamlit run app/streamlit_app.py```
or start the API:
```uvicorn app.main:app --reload```


## Datasets Used

- **Primary:** Amazon Reviews 2023 (McAuley Lab / Hugging Face) — metadata + reviews
- **Augmentation:** Retailrocket e-commerce sessions (Kaggle)
- **Subsampling strategy:** Electronics / Books categories for initial experiments


## Results & Benchmarks  
*(Preliminary / Expected – Update After Runs)*

### Model Performance

| Metric | Value (Target) | Notes / Hardware |
|------|---------------|------------------|
| Validation Perplexity | 18–28 | After 1–3 epochs on 500k samples |
| ROUGE-L (narrative coherence) | 0.32–0.45 | vs. reference human-like summaries |
| Human-rated relevance (1–5) | 3.8–4.3 / 5 | Blind eval on 100 samples |
| Inference time (per generation) | 400–900 ms | M2 Pro 16GB / RTX 3060 |
| Quantized model size (8-bit) | ~35–45 MB | bitsandbytes / torch.quantization |
| VRAM usage during fine-tuning | 6–10 GB | LoRA + bf16 + gradient checkpoint |
| Simulated CTR uplift vs random | +15–35% | Mock A/B on held-out sessions |

**MacBook-specific note:**  
Expect ~1.5–2× slower training than NVIDIA GPUs, but inference is very competitive thanks to MPS.  
Use the `--device mps` flag in training scripts if needed.

---

## Trade-offs & Design Decisions

| Aspect | Pro | Con | Chosen Mitigation / Rationale |
|------|-----|-----|-------------------------------|
| Small model (~60M params) | Fast inference, low memory, runs on laptop | Limited zero-shot knowledge | Heavy RAG + domain-specific fine-tuning |
| LoRA instead of full fine-tune | 100–1000× less VRAM, fast iterations | Slightly lower peak performance | Higher rank (32) + multi-epoch + good init |
| RAG integration | Grounds recs in real products, reduces hallucination | Adds ~100–300 ms latency | FAISS on CPU + precomputed embeddings + caching |
| Narrative vs simple list recs | Higher engagement, trust, conversion | Harder to evaluate automatically | Hybrid metrics (ROUGE + human + proxy CTR) |
| Tiktoken (GPT-2) tokenizer | Fast, proven, no training needed | Suboptimal for domain-specific tokens | Acceptable trade-off for speed & simplicity |
| Memmapped `.bin` files | Handles 100M+ tokens without RAM explosion | Slightly slower random access | Ideal for MacBook limited RAM (16–32GB typical) |
| No rotary embeddings (RoPE) | Simpler code, faithful to original notebook | Worse long-context extrapolation | Context ≤512 tokens sufficient |
| Streamlit + FastAPI | Quick interactive demo + production API | Not the fastest web framework | Ideal for portfolio showcase & rapid iteration |

---

## Future Extensions (Ideas)

- Multimodal: Condition on product images (CLIP / SigLIP embeddings)
- Session-based recs using Retailrocket click sequences
- On-device export (ONNX / Core ML for iOS/macOS)
- Preference optimization (DPO / ORPO) for more persuasive tone
- Real A/B testing harness with mock e-commerce backend

---

## License

MIT – feel free to fork, use, and adapt. Attribution appreciated.

---

## Author

**Abhishek** (Bengaluru, 2026)

Advanced portfolio project showcasing from-scratch SLM engineering + modern efficiency techniques.

⭐ Star if helpful!  
Questions, suggestions, or want to collaborate? Open an issue or PR.
