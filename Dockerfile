FROM python:3.7


WORKDIR /opt
COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 6002
ENV PORT 6002

ENTRYPOINT [ "python", "main.py", "--prod=True"]

