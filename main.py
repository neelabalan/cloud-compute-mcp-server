# server.py
from mcp.server.fastmcp import FastMCP
import pydantic
import requests
import logging
import yaml
import functools
import os
import typing


mcp = FastMCP("CloudComputeInfo")

# Configure logging
debug_env = os.getenv("DEBUG", "false").lower()
log_level = logging.DEBUG if debug_env in ("true", "1", "yes") else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(log_level)


class KeyedFilter:
    def __init__(self, key_path: str, separator: str = "."):
        # strip each segment so " foo . bar " -> ["foo","bar"]
        self.key_path = [p.strip() for p in key_path.split(separator)]
        logger.debug(f"filter considered - {key_path}")

    def filter(self, data: dict) -> dict:
        current = data
        keys = self.key_path
        try:
            # walk to the parent of the thing we want to delete
            for i in range(len(keys) - 1):
                if isinstance(current, dict):
                    current = current[keys[i]]
                elif isinstance(current, list):
                    current = current[int(keys[i])]
            # now delete the final key from the dict
            if isinstance(current, dict):
                current.pop(keys[-1], None)
            return data
        except (KeyError, IndexError, TypeError, ValueError):
            # if anything goes wrong just return data unchanged
            # might need to revisit this logic later
            return data


class FilterChain:
    def __init__(self, filters: list[KeyedFilter]):
        self.filters = filters

    def apply(self, data: dict | list) -> list | dict | None:
        if isinstance(data, list):
            return [self.apply(item) for item in data]
        elif isinstance(data, dict):
            for filter in self.filters:
                data = filter.filter(data)
            return data
        else:
            return None


class ComputeFilter(pydantic.BaseModel):
    vcpus_min: int
    vcpus_max: int
    price_max: float = float("inf")
    currency: str = "INR"
    partial_name_or_id: str | None = None
    architecture: typing.Literal["x86_64", "arm64"] = "x86_64"
    allocation: typing.Literal["ondemand", "spot"] = "ondemand"

    def __hash__(self):
        return hash((self.vcpus_min, self.vcpus_max, self.currency))


class AwsInstanceFilter(ComputeFilter):
    vendor: str = "aws"
    regions: str = "us-east-1"


class AzureVmFilter(ComputeFilter):
    vendor: str = "azure"
    regions: str = "eastus"


class GcpComputeFilter(ComputeFilter):
    vendor: str = "gcp"
    regions: str = "1230"


class Filters:
    top_level_keys = ["vendor", "vendor_id", "zone_id", "price_tiered", "price_upfront", "status", "zone", "observed_at"]

    region_keys = [
        "region.display_name",
        "region.country_id",
        "region.state",
        "region.address_line",
        "region.zip_code",
        "region.lon",
        "region.lat",
        "region.founding_year",
        "region.observed_at",
        "region.aliases",
        "region.api_reference",
        "region.country",
        "region.status",
        "region.green_energy",
        "region.vendor_id",
    ]

    server_keys = [
        # Very technical details not useful to everyone
        "server.server_id",
        "server.display_name",
        "server.score_per_price",
        "server.score",
        "server.cpu_flags",
        "server.cpus",
        "server.memory_ecc",
        "server.gpu_count",
        "server.gpu_memory_min",
        "server.gpus",
        "server.storage_type",
        "server.storages",
        "server.api_reference",
        "server.cpu_l1_cache",
        "server.cpu_l2_cache",
        "server.cpu_l3_cache",
        "server.cpu_manufacturer",
        "server.cpu_model",
        "server.cpu_family",
        "server.hypervisor",
        "server.observed_at",
        "server.min_price",
        "server.min_price_ondemand",
        "server.min_price_spot",
        "server.selected_benchmark_score",
        "server.selected_benchmark_score_per_price",
        "server.status",
    ]


def remove_none_values(data: dict | list) -> dict | list:
    if isinstance(data, dict):
        return {key: remove_none_values(value) for key, value in data.items() if value is not None}
    elif isinstance(data, list):
        return [remove_none_values(item) for item in data]
    else:
        return data


class ComputeInfoFetcher:
    filters_to_include = Filters.top_level_keys + Filters.region_keys + Filters.server_keys
    data = []
    filter_chain = FilterChain([KeyedFilter(_filter) for _filter in filters_to_include])
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "content-type": "application/json",
        "origin": "https://sparecores.com",
        "referer": "https://sparecores.com/",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "cross-site",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "x-application-id": "sc-www",
    }
    limit = 100

    @functools.lru_cache(maxsize=128)
    def fetch_data(self, filter: ComputeFilter) -> dict:
        api_url = "https://keeper.sparecores.net/server_prices"
        page = 1
        has_more_data = True
        total_response = []
        try:
            while has_more_data:
                params = {**filter.model_dump(), "page": page, "limit": self.limit}
                logger.debug(f"{params=}")
                response = requests.get(api_url, headers=self.headers, params=params)
                response.raise_for_status()
                if len(response.json()) == 0:
                    has_more_data = 0
                else:
                    page += 1
                    total_response.extend(response.json())
            logger.info(f"Pricing data fetched successfully. Total data - {len(total_response)}")
        except Exception as e:
            logger.error(f"Error fetching pricing data: {str(e)}")
        self.data = total_response

        return self.data

    def get_data(self) -> list:
        return self.data

    @functools.lru_cache(maxsize=128)
    def format(self) -> None:
        return yaml.dump(remove_none_values(self.data), default_flow_style=False)

    def _remove_duplicates(self, servers):
        seen = set()
        unique_servers = []
        for server in servers:
            # Create a tuple of the fields we want to check for duplicates
            identifier = (server["server_id"], server["price"], server["allocation"])
            if identifier not in seen:
                seen.add(identifier)
                unique_servers.append(server)
        return unique_servers


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Create a new instance only once for each class
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Ec2InstanceInfoFetcher(ComputeInfoFetcher, metaclass=Singleton):
    def fetch_data(self, filter: AwsInstanceFilter) -> dict:
        parent_data = super().fetch_data(filter)
        self.data = self._remove_duplicates(self.filter_chain.apply(parent_data))
        logger.info(f"Post filter total data size is {len(self.data)}")

        return self.data


# abstraction for the sake of it
# not really needed unless cloud specific data transformation required
class AzureVmInfoFetcher(ComputeInfoFetcher, metaclass=Singleton):
    def fetch_data(self, filter: AwsInstanceFilter) -> dict:
        parent_data = super().fetch_data(filter)
        self.data = self._remove_duplicates(self.filter_chain.apply(parent_data))
        logger.info(f"Post filter total data size is {len(self.data)}")

        return self.data


class GcpComputeEngineInfoFetcher(ComputeInfoFetcher, metaclass=Singleton):
    def fetch_data(self, filter: AwsInstanceFilter) -> dict:
        parent_data = super().fetch_data(filter)
        self.data = self._remove_duplicates(self.filter_chain.apply(parent_data))
        logger.info(f"Post filter total data size is {len(self.data)}")

        return self.data


# contextmanager?
ec2_instance_fetcher = Ec2InstanceInfoFetcher()
azure_vm_fetcher = AzureVmInfoFetcher()
gcp_compute_fetcher = GcpComputeEngineInfoFetcher()


@mcp.tool()
def ec2_instance_query(request: AwsInstanceFilter) -> str:
    "Get EC2 Instance details"
    ec2_instance_fetcher.fetch_data(request)
    return ec2_instance_fetcher.format()


@mcp.tool()
def azure_vm_query(request: AzureVmFilter) -> str:
    "Get Azure VM details"
    azure_vm_fetcher.fetch_data(request)
    return azure_vm_fetcher.format()


@mcp.tool()
def gcp_compute_query(request: GcpComputeFilter) -> str:
    "Get GCP Compute details"
    gcp_compute_fetcher.fetch_data(request)
    return gcp_compute_fetcher.format()


"""
Sample request

{
  "price_max": 20,
  "vcpus_max": 4,
  "vcpus_min": 4
}
"""
