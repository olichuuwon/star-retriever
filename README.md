# Star Retriever

Star Retriever is a comprehensive tool designed to interact with PDF documents using advanced natural language processing techniques. This project comprises several modules tailored for single PDF interactions, multi-PDF handling, and multimodal VLM (Vision-Language Model) parsing.

## Overview

- **Single PDF Chat**: Interact with individual PDF documents through a streamlined interface.
- **Multi-PDF Chat**: Process multiple PDFs simultaneously, facilitating collective query and discussion capabilities.
- **Multimodal VLM Parsing**: Leverages vision-language models for enhanced document interpretation. Currently, this component does not have a dedicated UI but can be executed and tested via command line or integration into existing scripts.

## Promising Developments

- **`langchain_multimodal.py`**: This script represents the most promising advancements in the project, integrating cutting-edge multimodal parsing with improved natural language interfaces.

## Vector Storage and Retrieval

- **ChromaDB**: Considered the future front-runner for vector storage solutions, though still under exploratory usage.
- **FAISS**: Currently implemented but requires further testing for optimal performance. Offers a robust solution but presents some challenges.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/olichuuwon/star-retriever.git
   cd star-retriever
   ```

2. **Install dependencies**:
   Ensure you have [pip](https://pip.pypa.io/en/stable/installation/) and Python environment setup correctly.
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**:
   Set up a `.env` file in the root directory to include any necessary API keys and configuration variables.

## Running the Application

To start the application, you can execute any of the following scripts depending on your use case:

- **Single PDF**:
  ```bash
  python -m streamlit run single_pdf.py
  ```

- **Multi-PDF**:
  ```bash
  python -m streamlit run multi_pdf.py
  ```

- **Multimodal VLM Parsing**:
  Currently, this module does not have a graphical interface. You can run it directly from the command line:
  ```bash
  python langchain_multimodal.py
  ```

## Project Structure

- `single_pdf.py`: Handles operations for single PDF interactions.
- `multi_pdf.py`: Designed to process and interact with multiple PDF documents in parallel.
- `langchain_multimodal.py`: Integrates VLMs for an advanced parsing experience, without a dedicated UI.

## Technologies and Tools

- **Streamlit**: Provides the web application framework.
- **LangChain**: Delivers the core language processing capabilities.
- **ChromaDB**: Under consideration for future vector DB implementations.
- **FAISS**: Currently used for vector storage but pending further testing.
