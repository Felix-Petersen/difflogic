import json
import time
import socket
import os


class ResultsJSON(object):

    def __init__(self, eid: int, path: str):
        self.eid = eid
        self.path = path

        self.init_time = time.time()
        self.save_time = None
        self.total_time = None

        self.args = None

        self.server_name = socket.gethostname().split('.')[0]

    def store_args(self, args):

        self.args = vars(args)

    def store_results(self, results: dict):

        for key, val in results.items():
            if not hasattr(self, key):
                setattr(self, key, list())

            getattr(self, key).append(val)

    def store_final_results(self, results: dict):

        for key, val in results.items():
            key = key + '_'

            setattr(self, key, val)

    def save(self):
        self.save_time = time.time()
        self.total_time = self.save_time - self.init_time

        json_str = json.dumps(self.__dict__)

        with open(os.path.join(self.path, '{:08d}.json'.format(self.eid)), mode='w') as f:
            f.write(json_str)

    @staticmethod
    def load(eid: int, path: str, get_dict=False):
        with open(os.path.join(path, '{:08d}.json'.format(eid)), mode='r') as f:
            data = json.loads(f.read())

        if get_dict:
            return data

        self = ResultsJSON(-1, '')
        self.__dict__.update(data)

        assert eid == self.eid

        return self


if __name__ == '__main__':

    r = ResultsJSON(101, './')

    print(r.__dict__)

    r.save()

    r2 = ResultsJSON.load(101, './')

    print(r2.__dict__)