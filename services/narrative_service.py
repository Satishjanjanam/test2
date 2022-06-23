"""narrative service"""
import json
import random
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

from string import Template

try:
    from cfuzzyset import cFuzzySet as FuzzySet
except ImportError:
    from fuzzyset import FuzzySet

from services.common import DateUtils
from utils.logging_handler import Logger
from services.constants import LOOKUP_DIMENSION
from services.anomaly_service import AnomalyDetector
from services.common import MetricUtils
from services.database_service import SQLDataBase

class RuleBasedNarrativeModel:
    """RuleBasedNarrativeModel class"""

    def __init__(self,
                 config):
        self.domain_path = config['domain_path']
        self.domain_config = self.load_domain(self.domain_path)

        self.database_creds = config['db_creds']
        Logger.info("Make a db connection")
        self.db_conn = SQLDataBase.connect_database(
            self.database_creds)

        self.db_schema = self.domain_config['db_schema']

        # create all the index
        self._create_index()

    def load_domain(self, domain_path):
        """function to load a domain"""
        with open(domain_path, 'r') as file:
            domain_config = json.loads(file.read())

        return domain_config

    def _create_index(self, ):
        """function to create a index"""
        Logger.info("Please wait while we start Creating Index...")

        # index all the metrics
        self.metric_index = FuzzySet()
        for dict_ in self.domain_config['metrics']:
            output = set([dict_['primary_metric_name']] + dict_['synonyms'])
            for m in output:
                self.metric_index.add(m.lower())

        # index all the dimensions
        cols = ','.join(self.db_schema['col_dimensions'])
        table_name = self.db_schema['table_name']
        query = f"SELECT DISTINCT {cols} FROM {table_name}"
        df_distinct = self.db_conn.execute_sql(query)

        self.campaign_index = FuzzySet()
        self.region_index = FuzzySet()
        self.country_index = FuzzySet()
        self.city_index = FuzzySet()
        self.product_index = FuzzySet()
        self.browser_index = FuzzySet()
        self.mobile_device_type_index = FuzzySet()
        self.referring_domain_index = FuzzySet()
        self.new_repeat_visitor_index = FuzzySet()

        for tup in df_distinct.values:
            self.campaign_index.add(tup[0].lower())
            self.region_index.add(tup[1].lower())
            self.country_index.add(tup[2].lower())
            self.city_index.add(tup[3].lower())
            self.product_index.add(tup[4].lower())
            self.browser_index.add(tup[5].lower())
            self.mobile_device_type_index.add(tup[6].lower())
            self.referring_domain_index.add(tup[7].lower())
            self.new_repeat_visitor_index.add(tup[8].lower())

        # create a synomys to metric mapping
        synonym2metric = {}
        # create metric to compare_metrics mapping
        metric2kpi = {}

        for dict_ in self.domain_config['metrics']:
            primary_metric_name = dict_['primary_metric_name']
            # handle kpi
            secondary_metrics = dict_['secondary_metric_name']
            compare_metrics = dict_['compare_metrics']
            kpi = []
            for col_name, name in zip(secondary_metrics, compare_metrics):
                kpi.append({
                    'name': name,
                    'col_name': col_name
                })
            metric2kpi[primary_metric_name] = kpi
            # handle synomys
            synonyms = dict_['synonyms']
            synonym2metric[primary_metric_name] = primary_metric_name
            for syn in synonyms:
                synonym2metric[syn] = primary_metric_name

        self.synonym2metric = synonym2metric
        self.metric2kpi = metric2kpi

        self.graph = self.domain_config['graph']
        #self.db_conn = None

        Logger.info("Sucessfully Created Index!")

    def validate_metric_dimension(self,
                                  metric_name,
                                  dimensions,
                                  threshold=0.4):
        """function to validate metric and dimension"""
        is_validated = True
        # first check metric name present in index or not otherwise ignore it !
        confidence, value = self.metric_index.get(metric_name)[0]
        if confidence >= threshold:
            metric_name = value
        else:
            metric_name = None

        entity_mapping = {}

        if metric_name is not None:
            metric = {
                "name": metric_name,
                "col_name": self.synonym2metric[metric_name]
            }

            if dimensions is not None:
                # if we have proper metric then look for dimensions
                for dict_ in dimensions:
                    col_name = dict_["col_name"]
                    value = dict_["value"]

                    if col_name == "campaign":
                        confidence, value = self.campaign_index.get(value)[0]
                        if confidence >= threshold:
                            dict_['value'] = value
                        else:
                            dict_['value'] = None
                    elif col_name == "region":
                        confidence, value = self.region_index.get(value)[0]
                        if confidence >= threshold:
                            # this is temperory
                            dict_['value'] = value.upper()
                        else:
                            dict_['value'] = None
                    elif col_name == "country":
                        value_ = value
                        confidence, value = self.country_index.get(value)[0]
                        if confidence >= threshold:
                            dict_['value'] = value
                        else:
                            dict_['value'] = None

                            # we didn't get match for dimension value in DB
                            if value_ in LOOKUP_DIMENSION[col_name]["examples"]:
                                dict_[
                                    'value'] = LOOKUP_DIMENSION[col_name]["synonym"]

                    elif col_name == "city":
                        confidence, value = self.city_index.get(value)[0]
                        if confidence >= threshold:
                            dict_['value'] = value
                        else:
                            dict_['value'] = None
                    elif col_name == "product":
                        confidence, value = self.product_index.get(value)[0]
                        if confidence >= threshold:
                            dict_['value'] = value
                        else:
                            dict_['value'] = None
                    elif col_name == "browser":
                        confidence, value = self.browser_index.get(value)[0]
                        if confidence >= threshold:
                            dict_['value'] = value
                        else:
                            dict_['value'] = None
                    elif col_name == "mobile_device_type":
                        confidence, value = self.mobile_device_type_index.get(value)[
                            0]
                        if confidence >= threshold:
                            dict_['value'] = value
                        else:
                            dict_['value'] = None
                    elif col_name == "referring_domain":
                        confidence, value = self.referring_domain_index.get(value)[
                            0]
                        if confidence >= threshold:
                            dict_['value'] = value
                        else:
                            dict_['value'] = None
                    elif col_name == "new_repeat_visitor":
                        confidence, value = self.new_repeat_visitor_index.get(value)[
                            0]
                        if confidence >= threshold:
                            dict_['value'] = value
                        else:
                            dict_['value'] = None
                # remove the entity whose value is None (i.e not present in index)
                dimensions = [dict_ for dict_ in dimensions if dict_[
                    'value'] is not None]

            entity_mapping = {
                "metric": metric,
                "dimensions": dimensions
            }
        else:
            is_validated = False

        return is_validated, entity_mapping

    def get_bm_narrative_template(
            self, metric_name):
        """function to get best match narrative template"""
        bm_id, bm_template = None, None
        # get the candidate templates
        candidate_templates = self.domain_config["narratives"]

        for dict_ in candidate_templates:
            valid_for = dict_['valid_for']
            if metric_name in valid_for:
                bm_id, bm_template = dict_['id'], dict_["template"]
                break

        if bm_template is None:
            options = [dict_ for dict_ in candidate_templates if dict_[
                'is_generic'] is True]

            if len(options) > 1:
                dict_ = random.choice(options)
                bm_id, bm_template = dict_['id'], dict_["template"]
            else:
                dict_ = options[0]
                bm_id, bm_template = dict_['id'], dict_["template"]

        return bm_id, bm_template

    def get_timeline(self, time_period):
        """function to get timeline"""
        grain = time_period['grain']
        from_ = time_period['from']
        to_ = time_period['to']

        timeline = None
        if grain == 'week':
            timeline = DateUtils.get_nlast_weeks(from_, to_)

        elif grain == 'month':
            timeline = DateUtils.get_nlast_months(from_, to_)

        elif grain == 'year':
            timeline = DateUtils.get_nlast_years(from_, to_)

        elif grain == 'days':
            timeline = DateUtils.get_nlast_days(from_, to_)

        elif grain == 'quarter':
            timeline = DateUtils.get_nlast_quarters(from_, to_)
        else:
            timeline = DateUtils.get_nlast_weeks(from_, to_)

        return timeline

    def get_data_from_db(self,
                         entity_mapping,
                         timeline,
                         limit=-1):
        """function to get data from database"""
        # re create the connection
        #self.db_conn = SQLDataBase.connect_database(
        #    self.database_creds)

        metric_col_name = entity_mapping['metric']['col_name']
        metric_name = entity_mapping['metric']['name']
        # fill spaces in the name
        if len(metric_name.split()) > 1:
            metric_name = '_'.join(metric_name.split())

        # time_period column
        time_series_col_name = self.db_schema['col_timeseries'][0]
        # col metrics
        col_metrics = self.db_schema['col_metrics']
        # comparable kpis
        metric_kpi = self.metric2kpi[metric_col_name]
        # name of the table
        table_name = self.db_schema['table_name']

        # range
        week_start_date_from = timeline[0]
        week_start_date_to = timeline[-1]

        # define a list of all conditions
        all_conditions = [
            f"({time_series_col_name}>='{week_start_date_from}' AND {time_series_col_name}<'{week_start_date_to}')"
        ]

        dimensions = []
        if entity_mapping['dimensions'] is not None:
            dim2val = {}
            for dict_ in entity_mapping['dimensions']:
                col_name = dict_['col_name']
                value = dict_['value']
                dimensions.append(col_name)

                if col_name not in dim2val:
                    dim2val[col_name] = [value]
                else:
                    dim2val[col_name].append(value)

            for dim, val in dim2val.items():
                # for a dimension with multiple values add OR condition
                condition = ' OR '.join(
                    [f"{dim}='{v}'" for v in val])
                condition = '(' + condition + ')'
                all_conditions.append(condition)

        all_conditions = list(set(all_conditions))
        all_conditions = ' AND '.join(all_conditions)

        # all columns
        all_cols = [time_series_col_name] + col_metrics + dimensions
        all_cols = list(set(all_cols))
        all_cols = ', '.join(all_cols)

        # execute the SQL query based on cols and conditions
        table_name = self.db_schema['table_name']
        query = None
        if limit > 0:
            query = f"SELECT {all_cols} FROM {table_name} WHERE {all_conditions} LIMIT {limit}"
        else:
            query = f"SELECT {all_cols} FROM {table_name} WHERE {all_conditions}"

        Logger.info(f"Run SQL query= {query}")
        df_sql_output = self.db_conn.execute_sql(query)

        data_timeline = []
        size = len(timeline) - 1
        for _iter in range(size):
            start_date = timeline[_iter]
            end_date = timeline[_iter + 1]
            df_sub = df_sql_output[(df_sql_output[time_series_col_name] >= start_date)
                                   & (df_sql_output[time_series_col_name] < end_date)]
            df_sub = df_sub[col_metrics]

            dict_ = dict(df_sub.sum(axis=0))
            for key, value in dict_.items():
                if key == "revenue":
                    dict_[key] = round(value, 2)
                else:
                    dict_[key] = int(value)

            # check if primary_metric is in dict_ or not
            if metric_col_name not in dict_:
                value = MetricUtils.calculate_extra_metric(
                    metric_col_name, dict_)
                dict_[metric_col_name] = value
            dict_[metric_name] = dict_[metric_col_name]

            # add all the kpi
            for kpi in metric_kpi:
                col_name = kpi['col_name']
                if col_name not in dict_:
                    value = MetricUtils.calculate_extra_metric(
                        col_name, dict_)
                    dict_[col_name] = value

            dict_[time_series_col_name] = DateUtils.format_date_to_str(
                start_date)
            data_timeline.append(dict_)

        if len(data_timeline) > 0:
            data_timeline = pd.DataFrame.from_records(data_timeline)

        #self.db_conn = None

        return data_timeline

    def predict_anomaly_narratives(self, data, entity_mapping, **kwargs):
        """function to predict anomalies with narratives"""
        # create the anomaly object
        anomaly_detector = AnomalyDetector(data)
        past_n = kwargs.get('past_n', 8)

        time_series_col_name = self.db_schema['col_timeseries'][0]
        metric_col_name = entity_mapping['metric']['col_name']

        # comparable kpis
        metric_kpi = self.metric2kpi[metric_col_name]
        all_metrics = [entity_mapping['metric']] + metric_kpi

        output = {}
        for metric_ in all_metrics:

            metric_col_name = metric_['col_name']
            metric_name = metric_['name']

            # for metric find anomly and template
            predictions = anomaly_detector.detect_anomaly(
                metric_col_name, time_series_col_name, past_n)
            # find the best template
            bm_id, bm_template = self.get_bm_narrative_template(
                metric_col_name)

            for dict_ in predictions:
                narrative, narrative_html = "", ""
                if dict_['is_anomaly'] is True:
                    narrative, narrative_html = NarrativeTemplate.get_narative_by_template(
                        dict_, metric_name,
                        time_series_col_name,
                        bm_template)
                dict_['narrative'] = narrative
                dict_['narrative_html'] = narrative_html

            # we have updated predictions for metric containing narrative
            output[metric_col_name] = predictions

        return output

    def prepare_response_payload(self,
                                 output, entity_mapping,
                                 is_validated=True):
        """function to get final payload"""
        if is_validated:
            metric_col_name = entity_mapping['metric']['col_name']
            metric_name = entity_mapping['metric']['name']

            compare_metrics = self.metric2kpi[metric_col_name]
            graph = self.graph

            response = {
                "code": 200,
                "anomaly": output,
                "metric": {
                    "name": metric_name.title(),
                    "col_name": metric_col_name
                },
                "compare_metrics": compare_metrics,
                "graph": graph
            }
        else:
            response = {
                "code": 428,
                "anomaly": [],
                "metric": None,
                "compare_metrics": None,
                "graph": None,
            }

        return response


class NarrativeTemplate:
    """NarrativeTemplate class"""
    @staticmethod
    def convert_placeholders_to_html(dict_placeholder):
        """helper function to convert placeholder to html"""
        dict_placeholder_html = {}
        for key, value in dict_placeholder.items():

            value_html = value

            if key in ["val_spike_drop"]:
                if value > 0:
                    # make it green
                    value_html = \
                        f"<span style='color:green;font-weight:bold;'>{value}</span>"
                elif value < 0:
                    # make it red
                    value_html = \
                        f"<span style='color:red;font-weight:bold;'>{value}</span>"
                else:
                    # make it yellow
                    value_html = \
                        f"<span style='color:yellow;font-weight:bold;'>{value}</span>"
            else:
                if key != "metric_above_below":
                    value_html = \
                    f"<span style='font-weight:bold;'>{value}</span>"

            dict_placeholder_html[key] = value_html

        return dict_placeholder_html

    @staticmethod
    def clean_placeholder(dict_placeholder):
        """helper function to clean placeholder"""
        # round off the float values if missed
        for key, value in dict_placeholder.items():
            if isinstance(value, (np.floating, float)):
                value = round(float(value), 3)
            else:
                if "_name" in key:
                    if isinstance(value, str):
                        if value.isupper() is False:
                            value = value.title()

            dict_placeholder[key] = value

        return dict_placeholder

    @staticmethod
    def helper_spike_drop(curr_value, prev_value):
        """helper function to find spike and drop"""
        spike_drop = None
        if prev_value > curr_value:
            spike_drop = '(drop)'
        elif prev_value < curr_value:
            spike_drop = '(spike)'
        prev_value_ = prev_value
        if prev_value_ == 0:
            prev_value_ = 1.0  # to avoid dividing by 0

        perc_change = round(
            ((curr_value - prev_value)/prev_value_ * 100.0), 2)

        return spike_drop, perc_change

    @staticmethod
    def helper_above_below(curr_value, prev_value):
        """helper function to find above and below"""
        above_below = None
        if curr_value > prev_value:
            above_below = 'above'
        else:
            above_below = 'below'

        return above_below

    @staticmethod
    def get_narative_by_template(prediction, metric_col_name,
                                 time_series_col_name, template):
        """ function to get narrative
            bm_id = (int) Id of the Narrative Template
            prediction = weekly prediction
            metric_name = name of metric
        """
        start_date = prediction[time_series_col_name]
        metric_val = prediction['value']
        sma_bound = prediction['sma_bound']

        metric_spike_drop, val_spike_drop = NarrativeTemplate.helper_spike_drop(
            metric_val, sma_bound)
        metric_above_below = NarrativeTemplate.helper_above_below(
            metric_val, sma_bound)

        dict_placeholder = {
            "start_date": start_date,
            "metric_val": metric_val,
            "metric_name": metric_col_name,
            "metric_spike_drop": metric_spike_drop,
            "metric_above_below": metric_above_below,
            'val_spike_drop': val_spike_drop,
        }
        # clean the placeholder
        dict_placeholder = NarrativeTemplate.clean_placeholder(
            dict_placeholder)
        dict_placeholder_html = NarrativeTemplate.convert_placeholders_to_html(
            dict_placeholder)

        template = Template(template)
        narrative = template.safe_substitute(
            **dict_placeholder)
        narrative_html = template.safe_substitute(
            **dict_placeholder_html)

        return narrative, narrative_html
