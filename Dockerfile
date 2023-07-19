# A Dockerfile is used to define how your code will be packaged. This includes
# your code, the base image and any additional dependencies you need.
# FROM waggle/plugin-base:1.1.1-ml
FROM python:3.8
# COPY requirements.txt /app/
# RUN python -m pip3 install --upgrade pip
# RUN pip3 install --upgrade setuptools pip

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt
RUN python3 -m pip install --upgrade pip --no-cache-dir
WORKDIR /
RUN pip install --upgrade pip

# RUN pip3 install ultralytics --no-cache-dir
RUN pip install ultralytics
RUN pip install opencv-python-headless
RUN pip install opencv-contrib-python-headless
RUN pip install py-cpuinfo

# Next, we include our code and specify what command should be run to execute it.
COPY app.py /app/
COPY utils/ /app/utils

# ADD https://web.lcrc.anl.gov/public/waggle/models/rideshare/best_s1.pt /app/best_s1.pt
ADD https://web.lcrc.anl.gov/public/waggle/models/rideshare/best_s1.pt /app/yolov8.pt

ADD https://web.lcrc.anl.gov/public/waggle/models/rideshare/best_s2.pt /app/best_s2.pt

## TESTING image
ADD test.jpg /app/test.jpg

# Finally, we specify the "main" thing that should be run.
WORKDIR /app
ENTRYPOINT [ "python3", "/app/app.py"]
