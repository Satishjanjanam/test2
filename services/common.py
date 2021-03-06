"""common utilities"""
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


class DateUtils:
    """DateUtils class"""
    @staticmethod
    def format_date_to_str(date, date_format='%Y-%m-%d'):
        """format date to str"""
        str_date = date.strftime(
            date_format) if date else None

        return str_date

    @staticmethod
    def format_str_to_date(str_date, date_format='%Y-%m-%d'):
        """format str to date"""
        date = datetime.strptime(
            str_date, date_format) if str_date else None

        return date

    @staticmethod
    def get_nlast_weeks(last_week, to_, past_n=24):
        """get nlast weeks"""
        last_week = datetime.strptime(last_week, '%Y-%m-%d')
        if to_ == "":
            curr_date = datetime.today().replace(
                hour=0, minute=0, second=0, microsecond=0)
            end_date = curr_date + timedelta(-curr_date.weekday())
        else:
            end_date = datetime.strptime(to_, '%Y-%m-%d')
        nweeks = [end_date, last_week]
        for i in range(past_n-1):
            past_date = last_week - timedelta(weeks=i+1)
            nweeks.append(past_date)

        nweeks = [week.date() for week in nweeks]
        nweeks.reverse()

        return nweeks

    @staticmethod
    def get_nlast_days(last_day, to_, past_n=24):
        """get nlast days"""
        last_day = datetime.strptime(last_day, '%Y-%m-%d')
        if to_ == "":
            end_date = datetime.today().replace(
                hour=0, minute=0, second=0, microsecond=0)
        else:
            end_date = datetime.strptime(to_, '%Y-%m-%d')
        ndays = [end_date, last_day]
        for i in range(past_n-1):
            past_date = last_day - timedelta(days=i+1)
            ndays.append(past_date)

        ndays = [day.date() for day in ndays]
        ndays.reverse()

        return ndays

    @staticmethod
    def get_nlast_months(last_month, to_, past_n=12):
        """get nlast months"""
        last_month = datetime.strptime(last_month, '%Y-%m-%d')
        if to_ == "":
            curr_date = datetime.today().replace(
                hour=0, minute=0, second=0, microsecond=0)
            end_date = ((curr_date.replace(day=1)) - (timedelta(days=0)))
        else:
            end_date = datetime.strptime(to_, '%Y-%m-%d')
        nmonths = [end_date, last_month]
        for i in range(past_n-1):
            past_date = last_month - relativedelta(months=i+1)
            nmonths.append(past_date)

        nmonths = [month.date() for month in nmonths]
        nmonths.reverse()

        return nmonths

    @staticmethod
    def get_nlast_years(last_year, to_, past_n=12):
        """get nlast years"""
        last_year = datetime.strptime(last_year, '%Y-%m-%d')
        if to_ == "":
            curr_date = datetime.today().replace(
                hour=0, minute=0, second=0, microsecond=0)
            end_date = curr_date.replace(day=1, month=1)
        else:
            end_date = datetime.strptime(to_, '%Y-%m-%d')
        nyears = [end_date, last_year]
        for i in range(past_n-1):
            past_date = last_year - relativedelta(years=i+1)
            nyears.append(past_date)

        nyears = [year.date() for year in nyears]
        nyears.reverse()

        return nyears

    @staticmethod
    def get_nlast_quarters(last_quarter, to_, past_n=12):
        """get nlast quarters"""
        last_quarter = datetime.strptime(last_quarter, '%Y-%m-%d')
        if to_ == "":
            quarter_number = round((last_quarter.month - 1) / 3 + 1)
            end_date = datetime(last_quarter.year, 3 * quarter_number + 1, 1)\
                + timedelta(days=0)
        else:
            end_date = datetime.strptime(to_, '%Y-%m-%d')
        nquarters = [end_date, last_quarter]
        for i in range(past_n-1):
            past_date = last_quarter - relativedelta(months=(i+1) * 3)
            nquarters.append(past_date)

        nquarters = [quarter.date() for quarter in nquarters]
        nquarters.reverse()

        return nquarters


class MetricUtils:
    """MetricUtils class"""
    @staticmethod
    def get_aov_metric(revenue, orders):
        """helper function to get aov metric"""
        aov = []
        for rev, ord_ in zip(revenue, orders):
            if int(ord_) == 0:
                aov.append(0.0)
            else:
                aov.append(round((rev/ord_), 2))

        return aov

    @staticmethod
    def get_conversion_rate_metric(orders, visits):
        """helper function to get conversion rate metric"""
        conversion_rate = []
        for ord_, vis in zip(orders, visits):
            if int(vis) == 0:
                conversion_rate.append(0.0)
            else:
                conversion_rate.append(round((ord_/vis) * 100.0, 4))

        return conversion_rate

    @staticmethod
    def get_conversion_metric(orders):
        """helper function to get the conversion metric"""
        return orders

    @staticmethod
    def get_upt_metric(units, orders):
        """helper function to get the upt metric"""
        upt = []
        for uni, ord_ in zip(units, orders):
            if int(ord_) == 0:
                upt.append(0)
            else:
                upt.append(int(uni/ord_))

        return upt

    @staticmethod
    def get_transaction_metric(orders):
        """helper function to get transaction metric"""
        return orders

    @staticmethod
    def get_cart_abandonment_rate_metric(cart_open, checkouts):
        """helper function to get cart abortment rate metric"""
        cart_abandonment_rate = []
        for cart, check in zip(cart_open, checkouts):
            if int(cart) == 0 or cart < check:
                cart_abandonment_rate.append(0.0)
            else:
                cart_abandonment_rate.append(
                    round(((cart - check)/cart) * 100.0, 2))

        return cart_abandonment_rate

    @staticmethod
    def get_cart_abandonment_metric(cart_open, checkouts):
        """helper function to get cart abandonment metric"""
        cart_abandonment = []
        for cart, check in zip(cart_open, checkouts):
            if int(cart) == 0 or cart < check:
                cart_abandonment.append(0)
            else:
                cart_abandonment.append(
                    int((cart - check)/cart))

        return cart_abandonment

    @staticmethod
    def get_variable_rate_metric(former, latter):
        """helper function to get variable_rate"""
        variable_rate = []
        for form, latt in zip(former, latter):
            if int(form) == 0:
                variable_rate.append(0.0)
            else:
                variable_rate.append(
                    round((latt/form) * 100.0, 2))

        return variable_rate

    @staticmethod
    def calculate_extra_metric(metric_name, dict_):
        """helper function to calculate extra metric"""
        if metric_name == 'aov':
            return MetricUtils.get_aov_metric(
                [dict_['revenue']], [dict_['orders']])[0]
        elif metric_name == 'conversion_rate':
            return MetricUtils.get_conversion_rate_metric(
                [dict_['orders']], [dict_['visits']])[0]

        elif metric_name == 'conversion':
            return MetricUtils.get_conversion_metric(
                [dict_['orders']])[0]

        elif metric_name == 'upt':
            return MetricUtils.get_upt_metric(
                [dict_['units']], [dict_['orders']])[0]

        elif metric_name == 'transaction':
            return MetricUtils.get_transaction_metric(
                [dict_['orders']])[0]

        elif metric_name == 'cart_abandonment_rate':
            return MetricUtils.get_cart_abandonment_rate_metric(
                [dict_['cart_open']], [dict_['checkouts']])[0]

        elif metric_name == 'cart_abandonment':
            return MetricUtils.get_cart_abandonment_metric(
                [dict_['cart_open']], [dict_['checkouts']])[0]
