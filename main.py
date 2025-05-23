from fastapi import FastAPI

from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

from llama_index.llms import LlamaCPP

app = FastAPI()

llm = LlamaCPP(

    model_path="./models/gemma-7b-it.Q4_K_M.gguf",

    context_window=2048,

    max_new_tokens=256,

    model_kwargs={"n_threads": 6}

)

documents = SimpleDirectoryReader("./docs").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(llm=llm)

class PromptRequest(BaseModel):

    prompt: str

@app.post("/query")

async def query_model(request: PromptRequest):

    response = query_engine.query(request.prompt)

    return {"answer": str(response)}