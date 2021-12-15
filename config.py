import os
import argparse
import json

parser = argparse.ArgumentParser(description='Discovery Anomaly Detection API')
parser.add_argument('--prod', type=bool, default=False, help='Input the Runtime Environment')
args = parser.parse_args()

dirname = os.path.dirname(__file__)

IS_DEVELOPMENT = not args.prod

with open('./configs/{}.json'.format('dev' if IS_DEVELOPMENT is True else 'prod')) as handle:
    env = json.loads(handle.read())

PORT = os.environ['PORT'] if os.environ.get('PORT') is not None else 6002

BASE_URL = env["base_url"]

SERVICE_ANOMALY_DETECTOR = env["service_anomaly_detector"]

STATIC_PATH = env['static']

CONFIG = {
    "domain_path": "./domain.json",
    "db_creds": {
        "database": "discovery", 
        "user": "postgres", 
        "password": "OZ@beEI*ecFp", 
        "host": "167.71.234.110", 
        "port": "5432",
        "reconnect": True
    }
}

