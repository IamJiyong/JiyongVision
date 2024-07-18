class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.logs = []

    def add_log(self, log, print_log=True):
        if print_log:
            print(log)
        self.logs.append(log)
    
    def add_config(self, config, print=True):
        self.add_log(log="Configurations:", print_log=print)
        for k, v in config.items():
            self.add_log(f"\t{k}: {v}", print_log=print)

    def save_logs(self):
        with open(self.filename, 'w') as file:
            for log in self.logs:
                file.write(log + '\n')
