FROM python:3.6
USER root
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y memcached cmake libgl1-mesa-glx libsm6 libxext6 libxrender-dev
RUN pip install --upgrade pip
RUN apt-get clean && rm -rf /tmp/* /var/tmp/*
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:5000","--timeout","120", "main:api"]
EXPOSE 5000
