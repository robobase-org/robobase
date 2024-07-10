import pytest
import multiprocessing


# Define a fixture to set the start method for multiprocessing
@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_start_method():
    # Set your desired start method here
    multiprocessing.set_start_method("forkserver")
