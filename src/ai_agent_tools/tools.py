import math


def calculator(expression: str):
    """
    Calculator tool.
    Evaluates basic mathematical expressions.
    """

    try:
        allowed = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
        }

        result = eval(expression, {"__builtins__": {}}, allowed)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def search(query: str):
    """
    Simulated search tool.

    In a real application, this could call
    Google, Bing, Tavily, SerpAPI, etc.
    """

    # Simulate a tool failure
    if "404" in query.lower():
        raise RuntimeError("Search tool failed: simulated 404 error")

    fake_results = {
        "tunisia": (
            "Tunisia is a country in North Africa. "
            "Its capital is Tunis."
        ),
        "python": (
            "Python is a high-level programming language "
            "widely used for AI, data science and web development."
        ),
    }

    query_lower = query.lower()

    for keyword, result in fake_results.items():
        if keyword in query_lower:
            return {
                "success": True,
                "result": result
            }

    return {
        "success": True,
        "result": f"No specific search result found for: {query}"
    }