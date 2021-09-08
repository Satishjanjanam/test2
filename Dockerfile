FROM python:3.7


WORKDIR /opt
COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5002
ENV PORT 5002

ENTRYPOINT [ "python", "main.py"]

