# 1. Start from a lightweight Python image
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /code

# 3. Copy requirements and install them
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. Copy the application code
COPY ./app /code/app

# 5. Run the application
# -w 4:  Start 4 Workers (Handling 4 requests at once)
# -k uvicorn.workers.UvicornWorker: Use FastAPI compatible workers
# --bind: Listen on port 80
CMD ["gunicorn", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:80"]