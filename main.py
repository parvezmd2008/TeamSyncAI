"""CLI entrypoint for the WhatsApp RAG Agent using LangChain and Gemini.

Usage examples:
  python main.py build --chat chat.txt
  python main.py query --question "What are the main topics discussed?"
  python main.py todos

Environment:
  Set GOOGLE_API_KEY to your Google API key for Gemini access.
"""

import argparse
import os
import sys

from rag_agent import build_index, load_vectorstore, create_qa_chain, extract_todos_from_chat
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="AI RAG Agent for WhatsApp chat analysis using LangChain and Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build vector index from chat
  python main.py build --chat chat.txt
  
  # Query the chat
  python main.py query --question "What should we do next?"
  
  # Extract todo items
  python main.py todos
  
  # Use custom Gemini model
  python main.py query --question "Summarize the key points" --model gemini-1.5-pro
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Build subcommand
    build_cmd = subparsers.add_parser("build", help="Build vector index from WhatsApp chat")
    build_cmd.add_argument("--chat", required=True, help="Path to " \
    "" \
    "" \
    " file")
    build_cmd.add_argument(
        "--persist",
        default="./chroma_db",
        help="Directory to persist Chroma database (default: ./chroma_db)",
    )
    build_cmd.add_argument(
        "--embedding-model",
        default="models/embedding-001",
        help="Google embedding model (default: models/embedding-001)",
    )
    
    # Query subcommand
    query_cmd = subparsers.add_parser("query", help="Query the chat analysis")
    query_cmd.add_argument(
        "--question",
        required=True,
        help="Question to ask about the chat",
    )
    query_cmd.add_argument(
        "--persist",
        default="./chroma_db",
        help="Directory where Chroma database is stored",
    )
    query_cmd.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    
    # Todos subcommand
    todos_cmd = subparsers.add_parser("todos", help="Extract action items from chat")
    todos_cmd.add_argument(
        "--persist",
        default="./chroma_db",
        help="Directory where Chroma database is stored",
    )
    todos_cmd.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)",
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key before running this program.")
        sys.exit(1)
    
    try:
        if args.command == "build":
            print("Building vector index from chat...")
            build_index(
                args.chat,
                persist_directory=args.persist,
                embedding_model=args.embedding_model,
            )
            print("✓ Index built successfully")
            
        elif args.command == "query":
            print("Loading chat...")
            chat_content = load_vectorstore(args.persist)
            print("Creating QA chain...")
            qa_chain = create_qa_chain(chat_content, gemini_model=args.model)
            print("Querying...\n")
            response = qa_chain.invoke(args.question)
            print("=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(response)
            
        elif args.command == "todos":
            print("Loading chat...")
            chat_content = load_vectorstore(args.persist)
            print("Extracting action items...\n")
            todos = extract_todos_from_chat(chat_content, gemini_model=args.model)
            if todos:
                print("=" * 60)
                print("ACTION ITEMS / TODO LIST:")
                print("=" * 60)
                for todo in todos:
                    print(todo)
            else:
                print("No action items found in the chat.")
        else:
            parser.print_help()
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
