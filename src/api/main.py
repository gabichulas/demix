from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def test():
    return {"hello": "world"}

@app.get("/items/{id_item}")
async def get_item(id_item: int):
    return {"item_id": id_item}

