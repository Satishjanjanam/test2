"""configuration"""
import os
import argparse
import json

parser = argparse.ArgumentParser(description='Discovery Anomaly Detection API')
parser.add_argument('--prod',
                    type=bool,
                    default=False,
                    help='Input the Runtime Environment')
args = parser.parse_args()

dirname = os.path.dirname(__file__)

IS_DEVELOPMENT = not args.prod

MODE = 'dev' if IS_DEVELOPMENT is True else 'prod'
with open(f'./configs/{MODE}.json', encoding="utf-8") as handle:
    env = json.loads(handle.read())

PORT = os.environ['PORT'] if os.environ.get('PORT') is not None else 5002

BASE_URL = env["base_url"]

SERVICE_ANOMALY_DETECTOR = env["service_anomaly_detector"]

STATIC_PATH = env['static']

CONFIG = {
    "domain_path": "./domain.json",
    "db_creds": {
        "database": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password":  os.environ["DB_PASSWORD"],
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
        "reconnect": True
    }
}
