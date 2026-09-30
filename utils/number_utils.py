def format_number(n):
    """Format number with K and M suffixes for human readability.
    
    Args:
        n (int): Number to format
        
    Returns:
        str: Formatted number string (e.g., "1.2K", "1.5M")
    """
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return str(n) 