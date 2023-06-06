# Use the official Python slim image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files to the container
COPY ./ ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port (8501)
EXPOSE 8501

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "Create_Summary.py"]
