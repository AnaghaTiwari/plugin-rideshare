# A Dockerfile is used to define how your code will be packaged. This includes
# your code, the base image and any additional dependencies you need.
FROM waggle/plugin-base:1.1.1-ml
COPY requirements.txt /app/
# RUN python -m pip3 install --upgrade pip
# RUN pip3 install --upgrade setuptools pip

RUN apt-get update \
    && apt-get install --no-install-recommends -y  \
        libgl1-mesa-glx libglib2.0-0 python3 python3-pip \
    && pip3 install ultralytics \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --no-cache-dir -r /app/requirements.txt
RUN pip3 install --upgrade pip

# Next, we include our code and specify what command should be run to execute it.
COPY app.py /app/
COPY utils/ /app/utils

ADD https://web.lcrc.anl.gov/public/waggle/models/rideshare/best_s1.pt /app/best_s1.pt
ADD https://web.lcrc.anl.gov/public/waggle/models/rideshare/best_s2.pt /app/best_s2.pt

# Finally, we specify the "main" thing that should be run.
WORKDIR /app
ENTRYPOINT [ "python3", "/app/app.py"]
