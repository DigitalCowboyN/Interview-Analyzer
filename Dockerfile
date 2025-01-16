# Use an official Python runtime as a parent image
FROM python:3.10
# You can choose any version that suits your script

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run Data_Preprocessor.py when the container launches
CMD ["python", "./Data_Preprocessor.py"]
