from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import shutil
import os
from typing import List
import predict as predict

app = FastAPI()

@app.post("/ok/")
async def check(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        file_path = f"input/{file.filename}"
        try:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            prediction = predict.get_prediciton(file_path)
            results.append({"filename": file.filename, "prediction": prediction})
        finally:
            print("done")
            #os.remove(file_path)
    return JSONResponse(content=jsonable_encoder(results))

#Command to run the file
#python3 -m uvicorn main:app --reload
