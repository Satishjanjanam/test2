"""main handler"""
from routes.base_handler import BaseHandler

class MainHandler(BaseHandler):
    """MainHandler class"""
    def get(self):
        return self.send_json(message="You have Successfully reached AnomalyDetection API")

