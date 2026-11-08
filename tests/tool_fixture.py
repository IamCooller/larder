"""Five tool schemas in the OpenAI function-calling format, with mixed argument types, for the forced-rows tests."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and country, e.g. Paris, France"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new event on the user's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "date": {"type": "string", "description": "Event date, YYYY-MM-DD"},
                    "start_time": {"type": "string", "description": "Start time, HH:MM 24h"},
                    "duration_minutes": {"type": "number", "description": "Duration in minutes"},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of attendee emails"},
                    "all_day": {"type": "boolean", "description": "Whether this is an all-day event"},
                },
                "required": ["title", "date", "start_time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search the local filesystem for files matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text or filename pattern"},
                    "extension": {"type": "string", "enum": ["pdf", "docx", "txt", "py", "any"], "description": "File extension filter"},
                    "max_results": {"type": "number", "description": "Maximum number of results to return"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a chat message to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Recipient username or email"},
                    "message": {"type": "string", "description": "Message body text"},
                    "urgent": {"type": "boolean", "description": "Mark the message as urgent"},
                },
                "required": ["recipient", "message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a basic arithmetic calculation between two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"], "description": "Operation to perform"},
                },
                "required": ["a", "b", "operation"],
                "additionalProperties": False,
            },
        },
    },
]

SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
