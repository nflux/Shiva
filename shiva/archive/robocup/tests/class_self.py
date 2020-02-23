class Swag:
    def __init__(self, message):
        self.message = message
    
    def print_message(self):
        print(self.message)

    def inst(self):
        this = self
        this.print_message()
        print(this.message)

if __name__ == "__main__":
    s = Swag('AndyShabandy')
    s.inst()