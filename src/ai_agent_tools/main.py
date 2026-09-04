from agent import Agent


def main():

    agent = Agent()

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = agent.run(user_input)

        print(f"Agent: {answer}")


if __name__ == "__main__":
    main()