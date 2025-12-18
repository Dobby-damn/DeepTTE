import datetime

class Logger:
    def __init__(self, exp_name):
        self.file = open('./logs/{}.log'.format(exp_name), 'w')

    def log(self, content):
        self.file.write(content + '\n')
        self.file.flush()
    def close(self):
        if not self.file.closed:
            self.file.close()


