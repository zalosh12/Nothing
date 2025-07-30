from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cls import Cls  # הקלאס שיצרת לטיפול במודל
import requests

app = FastAPI()
model_handler = Cls()


@app.on_event("startup")
def fetch_model_on_startup() :
    try :
        # כתובת הקונטיינר השני (לפי שם שירות ברשת docker או כתובת IP)
        response = requests.get("http://model-server:8000/export_model/")
        if response.status_code != 200 :
            raise Exception(f"Failed to fetch model: {response.text}")

        model_dict = response.json()
        model_handler.load_model(model_dict)

    except Exception as e :
        raise RuntimeError(f"Could not load model on startup: {e}")


class PredictRequest(BaseModel) :
    features: list


@app.post("/predict/")
def predict(request: PredictRequest) :
    try :
        result = model_handler.predict(request.features)
        return {"prediction" : result}
    except Exception as e :
        raise HTTPException(status_code=400, detail=str(e))

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from cls import Cls
#
# app = FastAPI()
# my_cls = Cls()
#
#
# class PredictInput(BaseModel):
#     features: list
#
#
#
# @app.post("/predict/")
# def predict(input_data: PredictInput):
#     try:
#         prediction = my_cls.predict(input_data.features)
#         return {"prediction": prediction}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
