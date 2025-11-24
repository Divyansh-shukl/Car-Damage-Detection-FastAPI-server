# server.py (replace existing /predict handler)
from fastapi import FastAPI, File, UploadFile
import traceback, logging, os
from model_helper import predict
app = FastAPI()
logger = logging.getLogger("uvicorn.error")

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_path = "temp_file.jpg"
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # log that a request arrived
        logger.info(f"Received file: {file.filename}, saved to {os.path.abspath(image_path)}")

        prediction = predict(image_path)
        return {"prediction": prediction}
    except Exception as e:
        # Log full traceback so Render logs show exactly what failed
        tb = traceback.format_exc()
        logger.error("Exception in /predict:\n" + tb)
        # Return a helpful error to client (avoid leaking sensitive info in prod)
        return {"error": str(e), "traceback": tb.splitlines()[-10:]}  # last 10 lines of traceback

