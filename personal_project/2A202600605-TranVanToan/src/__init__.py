"""Day 8 — RAG Pipeline v2: Pháp luật ma tuý & tin tức nghệ sĩ."""

# NumPy 2.0 compatibility monkey-patch for ChromaDB
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
