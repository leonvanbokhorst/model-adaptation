def test_imports():
    """Test all major package imports."""
    imports = [
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "fastapi",
        "sentence_transformers",
        "PIL",
        "sklearn",
        "plotly",
        "seaborn",
        "matplotlib",
        "networkx",
        "spacy",
        "langchain",
        "ollama",
        "openai",
    ]
    
    failed_imports = []
    for package in imports:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            failed_imports.append(f"✗ {package}: {str(e)}")
    
    if failed_imports:
        print("\nFailed imports:")
        for fail in failed_imports:
            print(fail)
    else:
        print("\nAll imports successful!")

if __name__ == "__main__":
    test_imports()
