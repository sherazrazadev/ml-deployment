import asyncio
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 1. Load Model
# Note: Sklearn is CPU only, but this logic prepares you for GPU/PyTorch
model = joblib.load("app/model.joblib")
class_names = ["setosa", "versicolor", "virginica"]

# 2. Define Schemas
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    prediction: str
    confidence: float
    batch_size_used: int  # Just to show you it's working!

# --- BATCHING ENGINE START ---
BATCH_SIZE = 32      # Max items to process at once
BATCH_TIMEOUT = 0.05 # Wait 50ms to collect items

queue = asyncio.Queue()

async def batch_processor():
    """Background task that runs forever, pulling from queue and predicting."""
    while True:
        # 1. Collect requests
        batch_data = []
        batch_futures = []

        # Get first item (wait if empty)
        data, future = await queue.get()
        batch_data.append(data)
        batch_futures.append(future)

        # Try to get more items immediately (up to BATCH_SIZE) without waiting
        # This is the "Dynamic" part. If 10 people click now, we grab all 10.
        try:
            while len(batch_data) < BATCH_SIZE:
                # wait only a tiny bit for more items
                data, future = await asyncio.wait_for(queue.get(), timeout=BATCH_TIMEOUT)
                batch_data.append(data)
                batch_futures.append(future)
        except asyncio.TimeoutError:
            pass # Timeout reached, process what we have

        if batch_data:
            try:
                # 2. PREDICT (The Vectorized "Layer B" Magic)
                # We stack list of inputs into one big Matrix
                features_batch = np.array(batch_data)

                # Run model ONCE for everyone
                probs = model.predict_proba(features_batch)
                preds = model.predict(features_batch)

                # 3. Distribute results back to waiting users
                for i, future in enumerate(batch_futures):
                    result = {
                        "class": class_names[int(preds[i])],
                        "confidence": float(np.max(probs[i])),
                        "batch_size": len(batch_data)
                    }
                    if not future.done():
                        future.set_result(result)
            except Exception as e:
                for future in batch_futures:
                    if not future.done():
                        future.set_exception(e)

# Start/Stop the background processor
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(batch_processor())
    yield
    task.cancel()

# --- BATCHING ENGINE END ---

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "High-Performance ML API"}

@app.post("/predict", response_model=IrisResponse)
async def predict(data: IrisRequest):
    # 1. Prepare data
    features = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]

    # 2. Create a "Future" (a ticket to get result later)
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # 3. Put in Queue and WAIT
    await queue.put((features, future))
    result = await future # This line pauses until batch_processor finishes

    # 4. Return
    return {
        "prediction": result["class"],
        "confidence": result["confidence"],
        "batch_size_used": result["batch_size"]
    }