import datetime

class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.logs = []

    def add_log(self, log, time=True, print_log=True):
        if time:
            log = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log}"
        if print_log:
            print(log)
        self.logs.append(log + '\n')
    
    def add_config(self, config, print_log=True):
        self.add_log(log="Configurations:", print_log=print_log)
        for k, v in config.items():
            self.add_log(f"\t{k}: {v}", time=False, print_log=print_log)

    def save_logs(self):
        with open(self.filename, 'w') as file:
            for log in self.logs:
                file.write(log + '\n')
