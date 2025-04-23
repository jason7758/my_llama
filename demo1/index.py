# **LlamaIndex** is a data framework for your LLM applications.
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 初始化Ollama LLM，使用Qwen2模型
llm = Ollama(model="qwen2:7b")

# 初始化本地嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# 设置全局LLM和嵌入模型
Settings.llm = llm
Settings.embed_model = embed_model

# 加载文档并创建索引
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎并执行查询
query_engine = index.as_query_engine()
# response = query_engine.query("什么是RAG")
response = query_engine.query("严灿平是谁")
print(response)