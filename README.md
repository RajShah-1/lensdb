# Traffic Video Pipeline

This project provides a modular pipeline for traffic and surveillance video analytics.  
It enables **video ingestion, frame sampling (frame-based or time-based), and embedding extraction** through a clean class-based interface.  
The design is extendable — you can plug in different embedders (e.g., CLIP, MobileCLIP) or sampling strategies, making it a solid foundation for research in compressed video indexing and learned database systems.

---

## 🔧 Setup

```bash
# 1. Create & activate a Conda environment
conda create -n lensdb python=3.10 -y
conda activate lensdb

# 2. Install Poetry inside the environment
pip install poetry

# 3. Install project dependencies
poetry install
```