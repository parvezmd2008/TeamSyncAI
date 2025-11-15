# WhatsApp RAG Agent with LangChain and Gemini

An AI agent that analyzes WhatsApp group chats using **LangChain** (latest version), **Google Gemini**, and **Chroma** vector database to perform Retrieval-Augmented Generation (RAG) for chat analysis and todo list generation.

## Features

- 📱 **WhatsApp Chat Parsing**: Automatically parse WhatsApp chat exports
- 🤖 **Gemini Integration**: Uses Google's latest Gemini models (gemini-pro, gemini-1.5-pro, etc.)
- 🔍 **RAG Pipeline**: Builds a vector store for semantic search and retrieval
- 📋 **Todo Extraction**: Automatically extract action items from chat discussions
- 💾 **Persistent Storage**: Stores embeddings in Chroma for fast retrieval
- 🔄 **Latest LangChain**: Built with LangChain 0.2+

## Prerequisites

- Python 3.10+
- Google API Key (for Gemini access)
- WhatsApp chat export file (as `.txt`)

## Installation

### 1. Clone or set up the project

```bash
cd d:\KMIT
```

### 2. Install dependencies

Using `requirements.txt`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or using `pyproject.toml`:

```powershell
python -m pip install -e .
```

### 3. Set up your Google API Key

```powershell
$env:GOOGLE_API_KEY = 'your-google-api-key-here'
```

Or permanently add to your PowerShell profile:

```powershell
[System.Environment]::SetEnvironmentVariable('GOOGLE_API_KEY', 'your-key-here', 'User')
```

## Usage

### Export WhatsApp Chat

1. Open WhatsApp on your phone
2. Go to a group chat
3. Tap the group name → More → Export chat
4. Choose "Without media"
5. Save as `chat.txt` in the project directory

### Build Vector Index

```powershell
python main.py build --chat chat.txt
```

This will:
- Parse your WhatsApp chat
- Create embeddings using Google's embedding model
- Store them in a Chroma database (`./chroma_db` by default)

### Query the Chat

Ask questions about your chat:

```powershell
python main.py query --question "What are the main topics discussed?"
```

### Extract Todo Items

Automatically extract action items from the chat:

```powershell
python main.py todos
```

## Advanced Usage

### Use a specific Gemini model

```powershell
python main.py query --question "Summarize the decision points" --model gemini-1.5-pro
```

Available models:
- `gemini-pro` (default, fastest)
- `gemini-1.5-pro` (more capable)
- `gemini-1.5-flash` (balanced)

### Specify custom vector store location

```powershell
python main.py build --chat chat.txt --persist D:\my_vectors
python main.py query --question "What happened?" --persist D:\my_vectors
```

### Use a different embedding model

```powershell
python main.py build --chat chat.txt --embedding-model models/embedding-001
```

## Project Structure

```
.
├── main.py              # CLI entrypoint
├── rag_agent.py         # Core RAG implementation
├── read_chat.py         # WhatsApp chat parser
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project configuration
├── chat.txt             # Your WhatsApp export (add this)
├── chroma_db/           # Vector store (auto-created)
└── README.md            # This file
```

## API Reference

### `read_chat.py`

```python
from read_chat import parse_whatsapp_chat

messages = parse_whatsapp_chat("chat.txt")
# Returns: [{"datetime": "...", "author": "...", "text": "..."}, ...]
```

### `rag_agent.py`

```python
from rag_agent import build_index, load_vectorstore, create_qa_chain, extract_todos_from_chat

# Build index
vectordb = build_index("chat.txt")

# Load existing index
vectordb = load_vectorstore("./chroma_db")

# Create QA chain
qa_chain = create_qa_chain(vectordb)
answer = qa_chain.run("Your question here")

# Extract todos
todos = extract_todos_from_chat(vectordb)
```

## Troubleshooting

### "GOOGLE_API_KEY environment variable not set"

```powershell
$env:GOOGLE_API_KEY = 'your-key'
```

### "Chat file not found"

Make sure `chat.txt` is in the current directory or provide the full path:

```powershell
python main.py build --chat C:\path\to\chat.txt
```

### Vector store not found

Ensure you've run the `build` command first:

```powershell
python main.py build --chat chat.txt
python main.py query --question "..."
```

### Slow response times

- Use `gemini-pro` (faster) instead of `gemini-1.5-pro`
- Reduce chat size by exporting only recent messages
- Increase `search_kwargs` in `create_qa_chain()` if you want more context

## Technology Stack

- **LangChain** - Orchestration framework for LLM applications
- **langchain-google-genai** - Google Gemini integration
- **Chroma** - Vector database for embeddings
- **Google Generative AI** - Gemini models and embeddings

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Your Google API key (required) | - |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON (alternative auth) | - |

## Notes

- Supports common WhatsApp chat export formats
- Handles multi-line messages automatically
- First run builds the index (takes a few seconds)
- Subsequent queries are fast (uses cached embeddings)
- Todos are extracted by asking Gemini to identify action items

## License

MIT

## Support

For issues, check:
1. API key is set correctly
2. Chat file is in proper WhatsApp export format
3. Internet connection is stable
4. LangChain and dependencies are up to date

