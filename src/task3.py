from langchain_core.runnables import RunnablePassthrough
passtrough = RunnablePassthrough()
result=passtrough.invoke("Hello World")
print(result)