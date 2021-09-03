import os
import tornado.web
from config import BASE_URL, SERVICE_ANOMALY_DETECTOR, IS_DEVELOPMENT, STATIC_PATH
from routes.error_handler import ErrorHandler
from routes.main_handler import MainHandler
from routes.anomaly_detector_handler import AnomalyDetectorHandler

def make_app():
    settings = {
        "default_handler_class": ErrorHandler,
        "debug": IS_DEVELOPMENT,
        "static_path": STATIC_PATH
    }
    api_path = {
        "service_anomaly_detector": SERVICE_ANOMALY_DETECTOR,
        "version": BASE_URL
    }
    return tornado.web.Application([
        (r"/", MainHandler),    
        ("/ping", MainHandler),
        ("/ping/", MainHandler),
        ("/{service_anomaly_detector}/{version}/detect".format(**api_path), AnomalyDetectorHandler),
        ("/{service_anomaly_detector}/{version}/detect/".format(**api_path), AnomalyDetectorHandler) 
    ], **settings
)

