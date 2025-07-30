from fastapi import FastAPI, HTTPException, Query
from fastapi import Request

from manager import ModelManager
from pydantic import BaseModel
import pickle
from typing import List, Dict, Union, Any
import base64
from fastapi.responses import JSONResponse


app = FastAPI()
manager = ModelManager()


class TrainRequest(BaseModel):
    url: str


@app.post("/train/")
def train_model(request: TrainRequest):
    try:
        accuracy =  manager.create_model_by_df(request.url)
        return {"message":"Model trained successfully","accuracy":accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


