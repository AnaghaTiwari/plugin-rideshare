# A Dockerfile is used to define how your code will be packaged. This includes
# your code, the base image and any additional dependencies you need.
FROM waggle/plugin-base:1.1.1-ml

WORKDIR /app

# Now we include the Python requirements.txt file and install any missing dependencies.
COPY requirements.txt .
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt

# Next, we include our code and specify what command should be run to execute it.
COPY main.py .
COPY test.py .
COPY test.jpg .

# Finally, we specify the "main" thing that should be run.
ENTRYPOINT [ "python3", "main.py"]
