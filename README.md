# Quantization Understanding

A project for exploring and understanding neural network quantization techniques.

---

## Setting Up a UV Project

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager. Follow these steps to initialize and work with this project using `uv`.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the installation:

```bash
uv --version
```

### 2. Initialize the Project

From the project root directory, run:

```bash
uv init
```

This creates a `pyproject.toml` file that describes your project and its dependencies.

### 3. Add Dependencies

Add packages your project needs:

```bash
uv add torch torchvision numpy matplotlib
```

To add a development-only dependency:

```bash
uv add --dev jupyter ipykernel
```

### 4. Sync the Environment

Install all dependencies into a local virtual environment (`.venv`):

```bash
uv sync
```

### 5. Run Scripts

Run a Python script inside the managed environment:

```bash
uv run python quantization.py
```

Or launch a Jupyter notebook:

```bash
uv run jupyter notebook
```

### 6. Activate the Virtual Environment (optional)

If you want to activate the environment directly:

```bash
source .venv/bin/activate
```

---

## Project Structure

```
.
├── pyproject.toml        # Project metadata and dependencies (created by uv init)
├── uv.lock               # Locked dependency versions (created by uv sync)
├── .venv/                # Virtual environment (created by uv sync)
├── quantization.py       # Main quantization script
├── quantization.ipynb    # Jupyter notebook for exploration
└── README.md
```
