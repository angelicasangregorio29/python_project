class Coffee:

    #initialize coffee with name and price 

    def __init__(self , name , price ):

        self.name = name

        self.price = price

class Order:

    #initialize order with empty list 


    def __init__(self):
        
        self.items = []

    # add coffee to order

    def add_item(self , coffee):

        self.items.append(coffee)

        print(f"Added {coffee.name} to your order")

    # calculate total price 

    def total(self):

        return sum(item.price for  item in self.items)

    # show order summary 

    def show_order(self):

        if not self.items:

            print("No items in order.")

            return

        print("\nYour Order:")

        for i , item in enumerate(self.items, 1):

            print(f"{i}. {item.name} - ${item.price}")

        print(f"Total: ${self.total()}\n")

    #handle checkout process 

    def checkout(self):                        