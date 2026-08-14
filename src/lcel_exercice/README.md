# LCEL Exercise

This project demonstrates how to use LangChain Expression Language (LCEL) to build a simple question-answering pipeline.

LCEL pipes (`|`) are used to connect the different steps of the pipeline. First, the question is sent to a prompt and an LLM to generate three sub-questions. Then, the sub-questions are parsed and sent to another prompt and LLM to generate answers. Finally, the answers are formatted and passed to a final prompt to produce a short summary.

The `batch()` method is used to process multiple sub-questions efficiently. Instead of calling the model separately for each sub-question, `batch()` sends multiple inputs to the same chain. This can reduce execution time and makes the application more efficient when several independent questions need to be answered.

The project also uses `RunnableLambda` to integrate custom Python functions into the LCEL pipeline.
