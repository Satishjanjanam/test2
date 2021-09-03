import json
from os.path import join, dirname
from jsonschema import validate

def load_json_schema(schema_name):
    relative_path = join('schemas', schema_name)
    absolute_path = join(dirname(__file__), relative_path)

    with open(absolute_path) as schema_file:
        return json.loads(schema_file.read())
    

def validate_request_anomaly_detector(func):
    async def validate_data(*args, **kwargs):
        try:
            data = json.loads(args[0].request.body)
            schema = load_json_schema("anomaly_detector_schema.json")
            validate(data, schema) 
            await func(*args, **kwargs)   
        except Exception as err:
            return args[0].send_json(status=500, message=str(err), success=False)
    return validate_data


