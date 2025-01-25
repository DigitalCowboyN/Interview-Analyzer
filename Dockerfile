# Use an official Python runtime as a parent image
FROM python:3.10
# You can choose any version that suits your script

# Set the working directory to /app
WORKDIR /app

# Install FFmpeg (latest version for compatibility with Mac)
# ffmpeg, sox, and libsndfile1 are commonly required
RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run Data_Preprocessor.py when the container launches
CMD ["python", "src/main.py"]
