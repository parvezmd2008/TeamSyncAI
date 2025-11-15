"""RAG Agent using LangChain and Google Gemini for WhatsApp chat analysis."""

import os
import time
from typing import List, Optional
from functools import wraps

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

from read_chat import parse_whatsapp_chat


def retry_with_exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying functions with exponential backoff on rate limit errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for rate limit or quota errors
                    if any(keyword in error_str for keyword in ['rate limit', 'quota', '429', '503']):
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            print(f"Rate limit hit. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                    raise
            return None
        return wrapper
    return decorator


class ResilientGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    """GoogleGenerativeAIEmbeddings wrapper with retry logic for rate limits."""
    
    def __init__(self, model: str = "models/embedding-001", google_api_key: str = "", max_retries: int = 3, **kwargs):
        super().__init__(model=model, google_api_key=google_api_key, **kwargs)
        self._max_retries = max_retries
    
    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with retry logic."""
        return super().embed_documents(texts)
    
    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    def embed_query(self, text: str) -> List[float]:
        """Embed query with retry logic."""
        return super().embed_query(text)


def build_documents_from_chat(chat_path: str) -> List[Document]:
    """Build LangChain Document objects from WhatsApp chat export."""
    msgs = parse_whatsapp_chat(chat_path)
    docs = []
    for msg in msgs:
        doc = Document(
            page_content=msg.get("text", ""),
            metadata={
                "author": msg.get("author", "Unknown"),
                "datetime": msg.get("datetime", "Unknown"),
            },
        )
        docs.append(doc)
    return docs


def build_index(
    chat_path: str,
    persist_directory: str = "./chroma_db",
    embedding_model: Optional[str] = None,
) -> str:
    """Load WhatsApp chat and return raw chat text for direct Gemini analysis.
    
    Args:
        chat_path: Path to chat.txt file
        persist_directory: Directory to persist (not used in direct mode)
        embedding_model: Not used in direct mode
        
    Returns:
        Raw chat text content
    """
    print(f"Loading chat file: {chat_path}")
    
    try:
        with open(chat_path, "r", encoding="utf-8") as f:
            chat_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Chat file not found: {chat_path}")
    
    # Parse to count messages
    msgs = parse_whatsapp_chat(chat_path)
    print(f"Loaded {len(msgs)} messages from chat")
    print(f"Chat size: {len(chat_content)} characters")
    
    return chat_content


def load_vectorstore(persist_directory: str) -> str:
    """Load the chat content directly.
    
    In direct mode, we don't use embeddings. This just loads the chat file.
    """
    # In direct mode, we'll load from the original chat.txt
    if not os.path.exists("chat.txt"):
        raise FileNotFoundError("chat.txt not found. Please run 'python main.py build --chat chat.txt' first.")
    
    with open("chat.txt", "r", encoding="utf-8") as f:
        return f.read()


def create_qa_chain(chat_content: str, gemini_model: Optional[str] = None):
    """Create a direct chat analysis chain using Gemini.
    
    Sends the entire chat directly to Gemini for analysis.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    model_name = gemini_model or "gemini-2.5-flash"
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,
    )
    
    # Custom prompt template for direct chat analysis
    prompt = PromptTemplate(
        input_variables=["chat_content", "question"],
        template="""Your name is TeamSyc, you are a productivity AI that helps users analyze WhatsApp group chats and make useful insights, summaries, and answer their questions.

You are analyzing the following WhatsApp group chat:

<chat>
{chat_content}
</chat>

Based on the chat history above, please answer the following question:

Question: {question}

Answer:""",
    )
    
    # Create a simple chain: prompt -> llm -> output_parser
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    # Return a wrapper that includes the chat content
    class ChatChain:
        def __init__(self, chain, chat_content):
            self.chain = chain
            self.chat_content = chat_content
        
        def invoke(self, question):
            return self.chain.invoke({"chat_content": self.chat_content, "question": question})
    
    return ChatChain(chain, chat_content)


def extract_todos_from_chat(chat_content: str, gemini_model: Optional[str] = None) -> List[str]:
    """Extract actionable todo items from chat using direct Gemini analysis."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    model_name = gemini_model or "gemini-2.5-flash"
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
    )
    
    prompt = PromptTemplate(
        input_variables=["chat_content"],
        template="""You are TeamSyc, a productivity AI assistant.

Analyze the following WhatsApp group chat and identify all action items, tasks, decisions, and things that need to be done:

<chat>
{chat_content}
</chat>

Please extract and list all actionable items from this chat. Format each item as a clear, actionable todo starting with a dash (-):

- Item 1
- Item 2
- etc.

If no action items are found, respond with: "No action items found in this chat."

Todo List:""",
    )
    
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke({"chat_content": chat_content})
    
    # Parse response into todo items
    todos = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line and (line.startswith("-") or line[0].isdigit()):
            todos.append(line)
    
    return todos if todos else ["No action items found in this chat."]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG Agent for WhatsApp chat analysis using Gemini"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build vector index from chat")
    build_parser.add_argument("chat", help="Path to chat.txt file")
    build_parser.add_argument(
        "--persist",
        default="./chroma_db",
        help="Directory to persist Chroma database",
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the vector store")
    query_parser.add_argument(
        "question",
        help="Question to ask about the chat",
    )
    query_parser.add_argument(
        "--persist",
        default="./chroma_db",
        help="Directory where Chroma database is stored",
    )
    query_parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use",
    )
    
    # Todos command
    todos_parser = subparsers.add_parser("todos", help="Extract todos from chat")
    todos_parser.add_argument(
        "--persist",
        default="./chroma_db",
        help="Directory where Chroma database is stored",
    )
    todos_parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use",
    )
    
    args = parser.parse_args()
    
    if args.command == "build":
        chat_content = build_index(args.chat, persist_directory=args.persist)
        print("\nChat loaded successfully!")
    elif args.command == "query":
        chat_content = load_vectorstore(args.persist)
        chain = create_qa_chain(chat_content, gemini_model=args.model)
        response = chain.invoke(args.question)
        print("\n" + "="*50)
        print("ANSWER:")
        print("="*50)
        print(response)
    elif args.command == "todos":
        chat_content = load_vectorstore(args.persist)
        todos = extract_todos_from_chat(chat_content, gemini_model=args.model)
        print("\n" + "="*50)
        print("TODO LIST:")
        print("="*50)
        for todo in todos:
            print(todo)
    else:
        parser.print_help()
