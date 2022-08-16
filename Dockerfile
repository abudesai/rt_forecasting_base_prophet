FROM python:3.8.0-slim as builder


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*


RUN pip install numpy==1.21.0
RUN pip install pandas==1.2.1
RUN pip install pystan==3.5.0
RUN pip install prophet==1.1
RUN pip install convertdate==2.4.0
RUN pip install lunarcalendar==0.0.9
RUN pip install holidays==0.14.2
RUN pip install tqdm==4.64.0


COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 


COPY app ./opt/app

WORKDIR /opt/app


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"



RUN chmod +x train &&\
    chmod +x test &&\
    chmod +x tune &&\
    chmod +x serve 