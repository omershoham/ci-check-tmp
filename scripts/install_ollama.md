# Installing the local LLM (Ollama)

The Stage-2 signal layer uses a local model so it costs zero Claude usage.
Any ~8 GB GPU comfortably runs a Q4-quantized 7B model.

## 1. Install Ollama

Option A — winget (Windows):
```powershell
winget install Ollama.Ollama
```

Option B — download the installer from https://ollama.com/download and run it.

After install, Ollama runs as a background service on `http://localhost:11434`.

## 2. Pull the model

```powershell
ollama pull qwen2.5:7b-instruct
```

(~4.7 GB download. Alternatives that also fit 8 GB: `llama3.1:8b-instruct-q4_K_M`,
`qwen2.5:7b`. If you change it, update `local_llm.model` in `config.yaml`.)

## 3. Verify

```powershell
ollama list                       # should show qwen2.5:7b-instruct
python -m src.local_llm           # prints a demo classification
```

If Ollama is not installed or not running, the pipeline still works — the signal
stage returns a neutral score and scoring proceeds without it.
