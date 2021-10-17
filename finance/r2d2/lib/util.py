class Color:
    def __init__(self):
        self.RED = '\033[31m'
        self.GREEN = '\033[32m'
        self.YELLOW = '\033[33m'
        self.END = '\033[0m'
    
    def red(self, text):
        print(self.RED + text + self.END)

    def yellow(self, text):
        print(self.YELLOW + text + self.END)

    def green(self, text):
        print(self.GREEN + text + self.END)