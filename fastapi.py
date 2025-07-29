from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stored_df = None

@app.post("/api/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    global stored_df
    contents = await file.read()
    with open("temp.xlsx", "wb") as f:
        f.write(contents)
    stored_df = pd.read_excel("temp.xlsx")
    return {"status": "ok"}

class Command(BaseModel):
    command: str

@app.post("/api/analyze")
async def analyze(cmd: Command):
    global stored_df
    if stored_df is None:
        return {"result": "请先上传 Excel 文件"}
    # 简单演示，根据指令返回列名
    if "列名" in cmd.command:
        return {"result": f"表格列名: {list(stored_df.columns)}"}
    else:
        return {"result": "暂未实现该指令，请扩展后端逻辑"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)