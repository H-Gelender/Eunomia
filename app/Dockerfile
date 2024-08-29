# Use the official Python 3.9 slim image as the base image
FROM python:3.9.19-slim-bullseye

# Upgrade pip to the latest version
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory on the host to the working directory in the container
COPY . .

# Install the required Python packages listed in the requirements.txt file
RUN pip install -r requirements.txt

# Expose port 8501 to allow communication with the Streamlit application
EXPOSE 8501

# Set the default command to run Streamlit with the application script
ENTRYPOINT ["streamlit", "run"]

# Specify the default application script to be run by Streamlit
CMD ["app.py"]
