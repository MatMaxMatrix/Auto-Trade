# TEST

import yaml
from abc import ABC

class Data(ABC):
    @classmethod
    def from_config(cls, config_path: str, *args) -> 'Data':
        with open(config_path, 'r') as f:
            conf = yaml.safe_load(f)
        data_config = conf['database']
        print(*args)
        print("^^^^^^^^")
        print(data_config)
        return cls(*args, **data_config)

class DatabaseConnection(Data):
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def connect(self):
        print(f"Connecting to {self.host}:{self.port} as {self.username}")

# Usage
db = DatabaseConnection.from_config('config.yaml')
db.connect()


