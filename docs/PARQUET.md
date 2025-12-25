Since you are training with LeRobot, you need two types of tools: high-level **CLI commands** (provided by LeRobot) for quick checks, and low-level **Python snippets** to inspect the raw Parquet data (the "hacker way" you prefer).

Here is your toolkit for inspecting and interacting with the dataset.

### 1. LeRobot Native Commands (CLI)

These are the "official" ways to check your data without writing code.

**A. Visualize the Replay (Rerun.io)**
This is the most important command. It launches a local web server to replay your episodes with video and telemetry synchronized.

```bash
# Visualize a dataset from the Hub (or local path)
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/aloha_sim_transfer_cube_human \
    --episode-index 0

```

* **What it does:** It reads the `videos/*.mp4` and aligns them with `data/*.parquet` to show you exactly what the robot "saw" and "felt."
* **Pro Tip:** Use this to verify that your `action` (motor command) doesn't lag behind your `observation` (video). Lag is the #1 killer of imitation learning policies.

**B. Check Data Statistics**
Before training, you must ensure your normalization stats (mean/std) look sane.

```bash
# Dumps the stats (mean, std, min, max) for every joint/dimension
cat meta/stats.json

```

* **Why check this?** If you see a `std` of `0.0` for a joint, that joint is dead/locked. If you see `max` values like `10^9`, your sensor data is corrupt.

### 2. Inspecting Parquet Directly (Python)

Since you want to understand the "first principles," here is how you bypass LeRobot and touch the raw Parquet files. This is useful for debugging specific artifact issues.

**Option A: The Pandas Way (Quick Look)**
Use this to check if your columns are named correctly.

```python
import pandas as pd

# Load one chunk of data (usually contains multiple episodes)
df = pd.read_parquet('data/chunk-000/file-000.parquet')

# View the first few rows (timestamps, joint positions)
print(df[['timestamp', 'observation.state', 'action']].head())

# Check the schema (ensure your dtypes are float32, not float64 to save RAM)
print(df.info())

```

**Option B: The DuckDB Way (Fast & Queryable)**
If your dataset is huge (e.g., 50GB+), Pandas might crash your RAM. [DuckDB](https://duckdb.org/) allows you to run SQL queries directly on the Parquet file without loading it.

```python
import duckdb

# Query: "Show me the max joint velocity in the first file"
# This runs instantly even on massive files
query = """
    SELECT MAX("observation.state"[1]) as max_joint_1 
    FROM 'data/chunk-000/file-000.parquet'
"""
print(duckdb.sql(query).show())

```

---

### 3. How Parquet and Arrow Interact in Imitation Learning

You asked about the interaction between **Parquet** and **Arrow**. This is the "secret sauce" of modern ML data loading.

In the context of LeRobot (and Hugging Face `datasets`), the workflow is:

**1. Disk (Parquet) → 2. RAM (Arrow) → 3. GPU (PyTorch Tensor)**

#### The Interaction

* **Parquet is the Storage Format (Compressed):** On your hard drive, the data is stored in Parquet. It uses heavy compression (Snappy/GZIP) and is column-oriented. You cannot "compute" on Parquet data; you must decompress it first.
* **Arrow is the Memory Format (Raw/Uncompressed):** When you load a dataset in LeRobot, the library (under the hood) reads the Parquet file and converts it into **Apache Arrow** format in RAM.
* **Memory Mapping (mmap):** This is the magic. Arrow tables can be "memory mapped." This means the OS treats the file on disk as if it were in RAM.
* **Zero-Copy (Almost):** If you access `dataset[500]`, the system jumps to the exact byte offset for row 500 without loading rows 0-499.



#### Why this matters for your Training (ACT / Diffusion Policy)

When you run your training loop:

1. **The Fetch:** The DataLoader requests a batch of indices (e.g., `[0, 10, 20]`).
2. **The Arrow Lookup:** Because Arrow is columnar, extracting the `action` column for these 3 rows is instant. It doesn't need to parse a JSON or CSV row-by-row.
3. **The Conversion:** Arrow arrays are memory-layout compatible with Numpy/PyTorch. Converting an Arrow array to a PyTorch tensor is extremely fast because they often share similar memory structures, minimizing CPU overhead.

**In summary:** Parquet saves your disk space (storage), while Arrow saves your CPU cycles (throughput). LeRobot handles the handshake between them so your GPU never starves waiting for data.