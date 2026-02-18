"""
Text formatting utilities for terminal display
Handles Arabic text reshaping for proper display
"""

import arabic_reshaper
from bidi.algorithm import get_display


def format_arabic_for_terminal(text: str, max_length: int = 50) -> str:
    """
    Format Arabic text for proper display in terminal
    
    Args:
        text: Arabic text to format
        max_length: Maximum length to display
        
    Returns:
        Properly formatted text for terminal display
    """
    
    if not text:
        return ""
    
    try:
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length - 3] + "..."
        
        # Reshape Arabic text (handles character joining)
        reshaped_text = arabic_reshaper.reshape(text)
        
        # Apply bidirectional algorithm (handles RTL)
        bidi_text = get_display(reshaped_text)
        
        return bidi_text
        
    except Exception as e:
        # Fallback: return length indicator
        return f"[Arabic text, {len(text)} chars]"


def format_query_preview(query: str, max_length: int = 50) -> str:
    """
    Create a preview of query for logging
    
    Args:
        query: Query text
        max_length: Maximum length
        
    Returns:
        Formatted preview
    """
    
    # Check if text contains Arabic characters
    has_arabic = any('\u0600' <= char <= '\u06FF' for char in query)
    
    if has_arabic:
        return format_arabic_for_terminal(query, max_length)
    else:
        # Non-Arabic text, just truncate
        if len(query) > max_length:
            return query[:max_length - 3] + "..."
        return query


def format_log_safe(text: str, max_length: int = 100) -> str:
    """
    Format any text safely for logging
    Handles Arabic, emojis, and special characters
    
    Args:
        text: Text to format
        max_length: Maximum length
        
    Returns:
        Safe text for terminal
    """
    
    try:
        # Try Arabic formatting first
        return format_arabic_for_terminal(text, max_length)
    except:
        # Fallback: ASCII-safe
        try:
            safe = text.encode('ascii', errors='replace').decode('ascii')
            if len(safe) > max_length:
                safe = safe[:max_length - 3] + "..."
            return safe
        except:
            return f"[{len(text)} chars]"