import os
import sys
sys.path.append(os.getcwd())

from config import CONFIG
from services.narrative_service import RuleBasedNarrativeModel
from utils.logging_handler import Logger

class AnomalyDetectorController(object):

    model = RuleBasedNarrativeModel(config=CONFIG)

    @staticmethod
    async def controller_method(req_payload):
        """
          a) Validate the metric/dimension using index 
          b) Get timeline for which results needs to be fetched
          d) Create a SQL query to fetch the data
          c) Predict anomaly and find the best match narrative template
          e) Finally, prepare the response payload 
        """
        # metric name
        metric_name = req_payload["metric_name"]
        # predicted time_period by NLU
        time_period = req_payload["time_period"]
        # predicted dimension by NLU
        dimensions = req_payload.get("dimensions", None)

        data_timeline = []
        entity_mapping = None

        Logger.info("\n")
        Logger.info("*"*100)

        # first validate the entities
        is_validated, entity_mapping = AnomalyDetectorController.model.validate_metric_dimension(
            metric_name, dimensions)
        
        Logger.info("is_validated= {}".format(is_validated))
        Logger.info("Primary Metric Mapping= {}".format(entity_mapping))
        Logger.info("*"*100)
        
        # if is validated flag is true then we are sure to get back results
        if is_validated is True:
            # get the time line from the time period asked by user
            timeline = AnomalyDetectorController.model.get_timeline(time_period)
            # get the data based on SQL query
            data_timeline = AnomalyDetectorController.model.get_data_from_db(
                                     entity_mapping=entity_mapping,
                                     timeline=timeline,
                                     limit=-1)
            output = AnomalyDetectorController.model.predict_anomaly_narratives(
                data_timeline, entity_mapping)

            Logger.info("*"*50)
  
        response =  AnomalyDetectorController.model.prepare_response_payload(
                             output,
                             entity_mapping,
                             is_validated)
        
        # # add a key as result 
        response_payload = req_payload
        response_payload['result'] = response 
        
        return response_payload
        