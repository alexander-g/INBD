FROM python:3.7.17-slim

ADD ./requirements_cpu.txt ./
RUN pip install -r requirements_cpu.txt

RUN apt-get update && apt-get install -y git
