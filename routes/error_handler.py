"""error handler"""
from routes.base_handler import BaseHandler

class ErrorHandler(BaseHandler):
    """ErrorHandler class"""
    def prepare(self):
        """error message prepare"""
        return self.send_json(
            message="The resource you are looking for is not found",
            status=404,
            success=False
        )