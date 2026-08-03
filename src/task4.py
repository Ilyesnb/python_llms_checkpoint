from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

llm = ChatOllama(model="llama3.2", temperature=0)

parser = StrOutputParser()

# 1. Sentiment chain
sentiment_prompt = ChatPromptTemplate.from_template("""
Analyze the sentiment of the following feedback.

Feedback:
{feedback}

Return only:
Positive
Neutral
Negative
""")

sentiment_chain = sentiment_prompt | llm | parser

# 2. Feedback type chain
feedback_type_prompt = ChatPromptTemplate.from_template("""
Feedback:
{feedback}

Sentiment:
{sentiment}

Classify into one category:
Complaint
Question
Suggestion
Praise

Return only the category.
""")

feedback_type_chain = feedback_type_prompt | llm | parser

# 3. Summary chain
summary_prompt = ChatPromptTemplate.from_template("""
Feedback:
{feedback}

Sentiment:
{sentiment}

Type:
{feedback_type}

Write one sentence summary.
""")

summary_chain = summary_prompt | llm | parser

workflow = (
    RunnableLambda(lambda feedback: {"feedback": feedback})
    | RunnableLambda(lambda x: {
        **x,
        "sentiment": sentiment_chain.invoke({"feedback": x["feedback"]})
    })
    | RunnableLambda(lambda x: {
        **x,
        "feedback_type": feedback_type_chain.invoke({
            "feedback": x["feedback"],
            "sentiment": x["sentiment"]
        })
    })
    | RunnableLambda(lambda x: {
        **x,
        "summary": summary_chain.invoke({
            "feedback": x["feedback"],
            "sentiment": x["sentiment"],
            "feedback_type": x["feedback_type"]
        })
    })
)
result = workflow.invoke(
    "The delivery was very late and the product arrived damaged. I would like a replacement."
)

print(result)
