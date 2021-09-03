import json
from tornado import web

class BaseHandler(web.RequestHandler):
    
    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Max-Age', 1000)
        self.set_header('Content-type', 'application/json')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('Access-Control-Allow-Headers',
                        'Content-Type, Access-Control-Allow-Origin, Access-Control-Allow-Headers, X-Requested-By, Access-Control-Allow-Methods')
    def options(self):
        pass
    def get(self):
        pass
    def post(self):
        pass

    def send_json(self, message="", success=True, response=None, status=200):
        self.set_status(status)
        return self.write(json.dumps({
            "success": success,
            "message": message,
            "response": response
        }))