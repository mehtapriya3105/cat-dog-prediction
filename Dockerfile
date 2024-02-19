# Use the official Python image as a base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

RUN pip install --upgrade pip

# Copy the requirements file into the container at /app
COPY requirement.txt /app/


# Install dependencies
RUN pip install --requirement /app/requirement.txt

RUN pip install fastapi uvicorn

RUN pip install python-multipart

# Copy the local codebase into the container at /app
COPY . /app/

# Expose port 8000 to the outside world
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]