import os
import pytest
import requests
from pathlib import Path

from keck_llm import get_base_directory


def pytest_configure(config):
    if not getattr(config.option, "check_links", False):
        return

    timeout = float(os.environ.get("CHECK_LINKS_TIMEOUT", "10"))
    original_request = requests.sessions.Session.request

    def request_with_timeout(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = request_with_timeout


@pytest.fixture
def small_text_path():
    return Path(get_base_directory()) / 'data' / 'small_data.txt'
