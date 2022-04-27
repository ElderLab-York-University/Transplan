class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Log(object):
    def __init__(self, message, bcolor, tag) -> None:
        self.message = message
        self.bcolor = bcolor
        self.tag = tag
    def __repr__(self) -> str:
        return f"{self.bcolor}{self.tag}:{bcolors.ENDC} {self.message}"

class WarningLog(Log):
    def __init__(self, message) -> None:
        super().__init__(message, bcolors.WARNING, "[Warning]")
