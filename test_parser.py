#!/usr/bin/env python3
"""Test script for WhatsApp chat parser."""

from read_chat import parse_whatsapp_chat
import json

# Test the parser
messages = parse_whatsapp_chat("chat.txt")
print(f"Successfully parsed {len(messages)} messages\n")

# Show all messages as JSON for clarity
for i, msg in enumerate(messages, 1):
    print(f"Message {i}:")
    print(f"  DateTime: {msg['datetime']}")
    print(f"  Author: {msg['author']}")
    print(f"  Text: {msg['text']}")
    print()
