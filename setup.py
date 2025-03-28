from setuptools import setup, find_packages

setup(
    name="notion_assistant",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "notion-client>=2.2.1",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.2",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "chromadb>=0.4.22",
        "sentence-transformers>=2.5.1",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
)
