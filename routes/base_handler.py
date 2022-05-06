"""base handler"""
import json
from tornado import web

class BaseHandler(web.RequestHandler):
    """BaseHandler class"""
    def set_default_headers(self):
        """Set default headers"""
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Content-type', 'application/json')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Access-Control-Allow-Headers',
                        'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')
    def options(self):
        """options method"""
        pass
    def get(self):
        """get method"""
        pass
    def post(self):
        """post method"""
        pass

    def send_json(self, message="", success=True, response=None, status=200):
        """helper function to send_json"""
        self.set_status(status)
        return self.write(json.dumps({
            "success": success,
            "message": message,
            "response": response
        }))