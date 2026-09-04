from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
model = ChatOllama(model="llama3.2", temperature=0.7)
def ask_question(question):
    response = model.invoke("give me one motivationalv quote for today")
    return response.content
graph = StateGraph(dict)
graph.add_node("start", "Start Node")
graph.set_entry_point("start")
graph.set_finish_point("start")
app = graph.compile()
result =app.invoke({})
print(result["output"])