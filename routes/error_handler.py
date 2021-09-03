from routes.base_handler import BaseHandler

class ErrorHandler(BaseHandler):
    def prepare(self):
        return self.send_json(
            message="The resource you are looking for is not found",
            status=404,
            success=False
        )