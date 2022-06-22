FROM python:3.7


WORKDIR /opt
COPY . .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# expect a build-time variable
ARG PORT=5002
# use the value to set the ENV var default
ENV PORT=$PORT 

ARG DB_NAME="discovery"
ENV DB_NAME=$DB_NAME

ARG DB_USER="postgres"
ENV DB_USER=$DB_USER

ARG DB_PASSWORD="OZ@beEI*ecFp"
ENV DB_PASSWORD=$DB_PASSWORD

ARG DB_HOST="discovery.postgres.database.azure.com"
ENV DB_HOST=$DB_HOST

ARG DB_PORT="5432"
ENV DB_PORT=$DB_PORT

EXPOSE 5002

ENTRYPOINT [ "python", "main.py", "--prod=True"]

