"""WhatsApp chat parser for extracting messages from exported chat.txt files."""

import re
from typing import List, Dict


def parse_whatsapp_chat(file_path: str) -> List[Dict[str, str]]:
    """
    Parse a WhatsApp chat export (text file) into a list of message dictionaries.
    
    Handles common WhatsApp export formats:
      - "12/31/20, 10:00 PM - Alice: Message text"
      - "2023-11-15, 14:03 - Bob: Another message"
      - "11/15/2025, 2:30:45 PM - Charlie: Message with timestamp"
      - System messages: "15/11/2025, 12:07 pm - Messages and calls are end-to-end encrypted."
    
    Args:
        file_path: Path to the WhatsApp chat.txt export file
        
    Returns:
        List of dicts with keys: 'datetime', 'author', 'text'
    """
    # Pattern for WhatsApp messages with author: date, time, author, message text
    user_pattern = re.compile(
        r"^(?P<datetime>[\d\/\-\.,\s:APMpm]+?)\s*-\s*(?P<author>[^:]+?):\s*(?P<text>.*)$"
    )
    
    # Pattern for system messages: date, time, message text (no author)
    system_pattern = re.compile(
        r"^(?P<datetime>[\d\/\-\.,\s:APMpm]+?)\s*-\s*(?P<text>[^:]+)$"
    )
    
    messages = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                
                # Try user message pattern first (with author)
                match = user_pattern.match(line)
                if match:
                    msg = {
                        "datetime": match.group("datetime").strip(),
                        "author": match.group("author").strip(),
                        "text": match.group("text").strip(),
                    }
                    messages.append(msg)
                else:
                    # Try system message pattern (no author)
                    match = system_pattern.match(line)
                    if match:
                        msg = {
                            "datetime": match.group("datetime").strip(),
                            "author": "System",
                            "text": match.group("text").strip(),
                        }
                        messages.append(msg)
                    else:
                        # Handle continuation lines (message text split across lines)
                        if messages:
                            messages[-1]["text"] += "\n" + line
    except FileNotFoundError:
        raise FileNotFoundError(f"Chat file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error parsing chat file: {e}")
    
    return messages


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python read_chat.py <path_to_chat.txt>")
        sys.exit(1)
    
    messages = parse_whatsapp_chat(sys.argv[1])
    print(f"Parsed {len(messages)} messages")
    for i, msg in enumerate(messages[:5]):
        print(f"\n[Message {i+1}]")
        print(f"  Author: {msg['author']}")
        print(f"  Time: {msg['datetime']}")
        print(f"  Text: {msg['text'][:100]}...")
