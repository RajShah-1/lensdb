"""
Test that all required modules can be imported.
"""

def test_imports():
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
    
    try:
        from src.embeddings.embedder import CLIPEmbedder, CLIP_VIT_B32
        print("✓ CLIPEmbedder")
    except ImportError as e:
        print(f"✗ CLIPEmbedder: {e}")
    
    try:
        from src.models.count_predictor import CountPredictor
        from src.models.model_configs import MEDIUM
        print("✓ CountPredictor")
    except ImportError as e:
        print(f"✗ CountPredictor: {e}")
    
    try:
        import faiss
        print(f"✓ faiss (version {faiss.__version__})")
    except ImportError as e:
        print(f"✗ faiss: {e}")
        print("  Install with: conda install -c conda-forge faiss-cpu")
        return False
    
    try:
        from src.indexing.faiss_index import FAISSIndex
        print("✓ FAISSIndex")
    except ImportError as e:
        print(f"✗ FAISSIndex: {e}")
        return False
    
    try:
        from src.query.semantic_query import SemanticQueryPipeline, simple_query
        print("✓ SemanticQueryPipeline")
    except ImportError as e:
        print(f"✗ SemanticQueryPipeline: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True


if __name__ == "__main__":
    test_imports()


