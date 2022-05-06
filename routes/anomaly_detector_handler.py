"""anomaly detector handler"""
import json
from routes.base_handler import BaseHandler
from controllers.anomaly_detector_controller import AnomalyDetectorController
from middleware.validation import validate_request_anomaly_detector


class AnomalyDetectorHandler(BaseHandler):
    """AnomalyDetectorHandler class"""
    def get(self):
        return self.send_json(
            status=405,
            success=False,
            message="Get method not allowed!",
            response=None)

    @validate_request_anomaly_detector
    async def post(self):
        """post method"""
        try:
            req_body = json.loads(self.request.body)
            return self.send_json(
                status=200,
                message="Response predicted successfully!",
                success=True,
                response=await AnomalyDetectorController.controller_method(
                    req_body)
                )
        except Exception as error:
            return self.send_json(
                status=500,
                success=False,
                message=str(error),
                response=None)
