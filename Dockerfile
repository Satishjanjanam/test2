FROM python:3.7


WORKDIR /opt
COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000
ENV PORT 5000

ENTRYPOINT [ "python", "main.py", "--prod=True" ]

