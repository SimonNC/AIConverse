import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory

load_dotenv()



chat = ChatOpenAI()

memory = ConversationSummaryMemory(
    #chat_memory = FileChatMessageHistory(file_path=os.path.join(os.path.dirname(__file__), "chat_history.json")),
    memory_key="messages",
    llm=chat,
    return_messages=True

)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">> ")
    result = chain.invoke(content)
    print(result["text"])
