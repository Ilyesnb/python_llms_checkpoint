from tools import calculator, search
from langchain_ollama import ChatOllama


class Agent:

    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

    def fallback(self, user_input):
        """
        Use the LLM directly when a tool fails
        or when no tool is necessary.
        """

        print("[ROUTE] FALLBACK → LLM")

        response = self.llm.invoke(user_input)

        return response.content

    def run(self, user_input):

        text = user_input.lower()

        # --------------------------------
        # 1. Calculator
        # --------------------------------

        math_keywords = [
            "+",
            "-",
            "*",
            "/",
            "sqrt",
            "sin",
            "cos",
            "tan"
        ]

        if any(keyword in text for keyword in math_keywords):

            print("[ROUTE] Calculator")

            # Remove natural-language instructions
            expression = text.replace("calculate", "")
            expression = expression.replace("what is", "")
            expression = expression.strip()

            try:
                result = calculator(expression)

                if result["success"]:
                    return f"Answer: {result['result']}"

                print("[TOOL ERROR] Calculator failed")

                return self.fallback(user_input)

            except Exception as e:

                print(f"[TOOL ERROR] Calculator exception: {e}")

                return self.fallback(user_input)

        # --------------------------------
        # 2. Search
        # --------------------------------

        search_keywords = [
            "search",
            "find information",
            "look up",
            "information about"
        ]

        if any(keyword in text for keyword in search_keywords):

            print("[ROUTE] Search")

            try:

                result = search(user_input)

                if result["success"]:
                    return f"Search result: {result['result']}"

                print("[TOOL ERROR] Search returned failure")

                return self.fallback(user_input)

            except Exception as e:

                print(f"[TOOL ERROR] Search exception: {e}")

                return self.fallback(user_input)

        # --------------------------------
        # 3. No tool needed
        # --------------------------------

        print("[ROUTE] Direct LLM")

        return self.fallback(user_input)