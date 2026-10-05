<div align="center">

<h1 align="center">HiFo-Prompt</h1>

<h3 align="center">Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design</h3>

<p align="center">
<strong>🧠 Hindsight Insight Pool</strong> · <strong>🔭 Foresight Evolutionary Navigator</strong> · <strong>🔄 Closed-Loop Evolution</strong>
</p>

![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Release](https://img.shields.io/badge/Release-v1.0.0-green?style=flat-square)

[English](./README.md) · [中文 (Chinese)](./README_CN.md)

</div>

<br/>

Welcome! This repository provides the code implementation for the paper [HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design](https://arxiv.org/abs/2508.13333).

## 📢 News

- **Jan. 2026**: 🎉🎉 HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design has been accepted at **ICLR 2026** as a Poster!
<br/>


## 📖 Introduction

<div align="center">
  <img src="./assets/framework.jpg" width="100%">
  <p><em>The framework of HiFo-Prompt</em></p>
</div>

**HiFo-Prompt** (Hindsight-Foresight Prompt) is a novel framework for **Automatic Heuristic Design (AHD)** that synergizes **Large Language Models (LLMs)** with **Evolutionary Computation (EC)**.

Existing LLM-based methods often suffer from *short-term memory* (forgetting successful tricks) and *lack of direction* (randomly searching without a strategy). HiFo-Prompt solves this by introducing two key mechanisms:

- **🧠 Hindsight (The Insight Pool)**: A self-evolving knowledge base that distills and stores "design principles" from high-performing heuristics, preventing the system from reinventing the wheel.

- **🔭 Foresight (The Evolutionary Navigator)**: A meta-controller that monitors population dynamics (stagnation, diversity) and actively switches search regimes (*Explore*, *Exploit*, or *Balance*) via specific **Design Directives**.

<br/>

<div align="center">
  <img src="assets/example.jpg" width="100%">
  <p><em>Dynamic prompt generation process of HiFo-Prompt</em></p>
</div>

## 🔥 Key Features

| Component | Function | Why it matters |
|-----------|----------|----------------|
| **Insight Pool** | Extracts & Reuses Knowledge | Instead of discarding parents, we extract *why* they worked. The prompts are augmented with proven "Insights". |
| **Evolutionary Navigator** | Adaptive Control | Detects if the search is stuck (stagnation) or too narrow (low diversity) and dynamically adjusts the prompt strategy. |
| **Decoupled Evaluation** | Efficient Pipeline | Decouples "Thought" from "Code", allowing for faster iteration and lower token consumption compared to standard methods. |

<br/>

## 🛠️ Installation

We recommend using **Conda** to manage the environment.

```bash
# 1. Create environment
conda create -n hifo python=3.10
conda activate hifo

# 2. Clone repository
git clone https://github.com/Challenger-XJTU/HiFo-Prompt.git
cd HiFo

# 3. Install dependencies
cd hifo
pip install .
```

<br/>

## 🚀 Quick Start

> **Note**: You must have an LLM API key (e.g., OpenAI, DeepSeek, Qwen) or a local LLM server running.

### 1. Basic Usage Structure

```python
from hifo import hifo
from hifo.utils.getParas import Paras

# 1. Initialize Parameters
paras = Paras() 

# 2. Configure HiFo
paras.set_paras(
    method = "hifo",               
    problem = "tsp_construct",          # Problem: 'tsp_construct', 'bp_online'
    llm_api_endpoint = "https://api.deepseek.com/chat/completions", # Your API Endpoint
    llm_api_key = "sk-xxxxxxxx",        # Your API Key
    llm_model = "deepseek-flash",       # Model Name
    ec_pop_size = 8,                    # Population size
    ec_n_pop = 8,                      # Number of generations
    exp_n_proc = 4,                     # Parallel threads for evaluation
    exp_debug_mode = False              # Set True to see prompt construction details
)

# 3. Initialize & Run
evolution = hifo.EVOL(paras)
evolution.run()
```

### 2. Running Examples

We provide ready-to-run scripts for standard combinatorial optimization problems.

#### Traveling Salesman Problem (TSP)

Constructive heuristic design for TSP.

```bash
cd examples/tsp_construct
python runHiFo.py
```

#### Online Bin Packing (BPP)

Designing scoring functions for online packing.

```bash
cd examples/bp_online
python runHiFo.py
```

#### Custom Problem

```bash
cd examples/user_XXX
python runHiFo.py
```

<br/>

## ⚙️ LLM Configuration

HiFo-Prompt supports both **remote APIs** and **local LLM deployment**.

### Option A: Remote API (Recommended)

Supported protocols: OpenAI-compatible APIs (DeepSeek, Moonshot, ChatGPT, Gemini, etc.).

`llm_api_endpoint` accepts either a bare host or a full URL. Prefer the **full URL**
so the path is explicit; a bare host falls back to Aliyun's
`/compatible-mode/v1/chat/completions`, which only fits DashScope-style providers.

Modify `runHiFo.py`:

```python
# DeepSeek
llm_api_endpoint = "https://api.deepseek.com/chat/completions"
llm_api_key = "sk-xxxxxxxx"
llm_model = "deepseek-flash"
```

```python
# OpenAI (bare host, uses the default chat-completions path)
llm_api_endpoint = "api.openai.com"
llm_api_key = "your_key"
llm_model = "gpt-4o"
```

For providers whose OpenAI-compatible path differs, pass the full URL:

```python
llm_api_endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
llm_api_key = "your_key"
llm_model = "gemini-2.5-flash"
```

> **Note**: This endpoint must be an OpenAI-compatible chat-completions URL.
> Provider-native endpoints (e.g. Gemini's `/v1beta/models/<model>:generateContent`)
> use a different request/response schema and are not supported.

> **Note on quotas**: HiFo issues hundreds of LLM requests per run
> (`ec_pop_size x len(ec_operators) x ec_n_pop`, plus the initial population).
> Free tiers often allow only a few dozen requests per model per day, which
> causes HTTP 429 errors mid-run. If that happens, the exact API error is now
> printed; pick a model with enough quota or enable billing.

### Option B: Local LLM (vLLM / HuggingFace)

1. Start your local server (e.g., using vLLM):

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

2. Configure HiFo:

```python
llm_use_local = True
llm_local_url = "http://localhost:8000/v1/chat/completions"
```

<br/>

## 📂 Project Structure

```
HiFo-Prompt/
├── hifo/
│   ├── src/hifo/
│   │   ├── methods/
│   │   │   └── hifo/
│   │   │       ├── hifo.py                   # Main HiFo Algorithm
│   │   │       ├── hifo_evolution.py         # Evolution Operators (i1, e1, m1, etc.)
│   │   │       ├── insight_pool.py           # 🧠 Hindsight Module
│   │   │       └── evolutionary_navigator.py # 🔭 Foresight Module (Regime Control)
│   │   ├── llm/                              # LLM Interfaces
│   │   ├── problems/                         # Problem Definitions
│   │   └── utils/                            # Parameter parsing & helpers
│   └── setup.py
├── examples/                                 # Problem-specific runners (TSP, BPP, etc.)
└── docs/                                     # Documentation & Tutorials
```

<br/>

## 📊 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Algorithm method (`hifo`, `ael`) | `hifo` |
| `problem` | Problem type | `tsp_construct` |
| `ec_pop_size` | Population size per generation | `8` |
| `ec_n_pop` | Number of generations | `8` |
| `exp_n_proc` | Number of parallel processes | `4` |
| `eva_timeout` | Evaluation timeout (seconds) | `300` |
| `exp_debug_mode` | Enable debug output | `False` |

<br/>

## 📜 Citation & Community 

We are actively maintaining HiFo-Prompt and thrilled to hear from the community!

- Need Help? If you run into any bugs or have feature requests, please check the Issues page or submit a new one.
- Collaboration: We are open to discussions on AHD and LLMs. Feel free to reach out via email or pull requests. Let's push the boundaries of automated algorithm design together! 🤝

Support Us: If HiFo-Prompt aids your research or if you like our approach, please Star ⭐ or Fork 🍴 this repository. Your support drives our updates!

```bibtex
@inproceedings{chen2026hifo,
  title     = {HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design},
  author    = {Chen, Chentong and Zhong, Mengyuan and Fan, Ye and Shi, Jialong and Sun, Jianyong},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=imSLzfZ6av}
}

```

<br/>

## ✨ Acknowledgments  

Our work builds upon [EoH](https://github.com/FeiLiu36/EoH) and [ReEvo](https://github.com/ai4co/reevo), and we thank the authors for their inspiring work. We also acknowledge [LLM4AD](https://github.com/Optima-CityU/LLM4AD) and [FM4CO](https://github.com/ai4co/awesome-fm4co) for their valuable learning resources on LLM-based AHD.

<div align="center">

**[⬆ Back to Top](#hifo-prompt)**

</div>
