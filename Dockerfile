# ✅ Use an official Python base image
FROM python:3.10

# ✅ Set the working directory
WORKDIR /app

# ✅ Copy everything into the container
COPY . .

# ✅ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
