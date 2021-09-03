import psycopg2
import pandas as pd


class SQLDataBase(object):
    """Class to connect to the SQL DataBase (Postgres)"""

    def __init__(self, config, conn):
        self.config = config
        self.conn = conn
    
    @classmethod
    def connect_database(cls, config):
        """Get database"""

        conn = psycopg2.connect(
                database=config["database"], 
                user = config["user"], 
                password = config["password"], 
                host = config["host"], 
                port = config["port"])

        return cls(config, conn)
    
    def execute_sql(self, sql_query):
        """function to execute the SQL query"""
        data = pd.read_sql_query(sql_query, self.conn, 
                                parse_dates=['refresh_date'])

        return data
