import requests

API_BASE_URL = "http://127.0.0.1:8000"


def get_api_data(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as error:
        return {
            "error": str(error)
        }


def get_summary(filters=None):
    return get_api_data("/analytics/summary", filters)


def get_metadata():
    return get_api_data("/metadata")


def get_filters():
    return get_api_data("/filters")


def get_sales_trend(filters=None, granularity="month"):
    params = filters.copy() if filters else {}
    params["granularity"] = granularity

    return get_api_data("/analytics/sales-trend", params)


def get_products(filters=None, limit=10):
    params = filters.copy() if filters else {}
    params["limit"] = limit

    return get_api_data("/analytics/products", params)


def get_countries(filters=None, limit=50):
    params = filters.copy() if filters else {}
    params["limit"] = limit

    return get_api_data("/analytics/countries", params)


def get_customers(filters=None, limit=20):
    params = filters.copy() if filters else {}
    params["limit"] = limit

    return get_api_data("/analytics/customers", params)