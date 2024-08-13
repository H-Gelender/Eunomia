import json

from nodes import RAGNode, PreprocessQuestionNode, ChatNode, ChoosePathNode
from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from typing import Dict, TypedDict, Optional

class GraphState(TypedDict):
    question: str = ""
    preproccess_question: str = ""
    books: str = ""
    documents: str = ""
    ids: str = ""
    answer: str = ""
    chat_history: str = ""
    path: str = ""

class EunomiaGraph:

    def __init__(self, llm, embeddings):

        self.embeddings = embeddings
        self.llm = llm
        self.workflow = StateGraph(GraphState)
        self.app = self.init_graph()

        self.ids = []
        self.books = []
        self.documents = []

    def init_node(self):
        self.rag_node = RAGNode(self.llm, "eunomia", self.embeddings)
        preprocess_node = PreprocessQuestionNode(self.llm)
        chat = ChatNode(self.llm)
        choose_path = ChoosePathNode(self.llm)

        self.workflow.add_node("ChoosePath_node", choose_path.run)
        self.workflow.add_node("Preprocess_node", preprocess_node.run)
        self.workflow.add_node("RAG", self.rag_node.run)
        self.workflow.add_node('Retriever_node', self.retriever_node)
        # self.workflow.add_node('Final_Node', self.final_node)
        self.workflow.add_node("Chat_node", chat.run)

    def init_edges(self):

        self.workflow.set_entry_point("ChoosePath_node")
        self.workflow.add_conditional_edges(
            "ChoosePath_node",
            self.path,
            {
                "Preprocess_node": "Preprocess_node",
                "Chat_node": "Chat_node"
            }
        )
        self.workflow.add_edge("Preprocess_node", "RAG")
        self.workflow.add_edge("RAG", "Retriever_node")
        self.workflow.add_edge("Retriever_node", "Chat_node")
        self.workflow.add_edge("Chat_node", END)
        # self.workflow.add_edge("Final_Node", END)

    def init_graph(self):
        self.init_node()
        self.init_edges()
        return self.workflow.compile()

    def retriever_node(self, state:dict):
        ids = []
        books = []
        documents = []

        question = state.get("question", "")["question"].strip()
        docs = self.rag_node.retriever.invoke(question)

        for doc in docs:
            ids.append(doc.metadata["id"])
            books.append(doc.metadata["book"])
            documents.append(doc.page_content)

        self.ids = ids
        self.books = books
        self.documents = documents

        return {"ids" : ids,"books" : books,"documents" : documents }

    def path(self, state:dict):
        path = state.get("path", "")
        return path

    def run(self, question):
        inputs = {"question": question}
        result = self.app.invoke(inputs)
        return result["answer"]
