import random
print("Welcome to the Two Dice Roller!")


class Dice:
    def __init__(self):
        self.dice1 = 0
        self.dice2 = 0

    def roll(self):
        self.dice1 = random.randint(1, 6)
        self.dice2 = random.randint(1, 6)


playing = True
while playing:
    choice = input("Do you want to roll the dice? (y/n): ").lower()
    if choice == "y":
        dice = Dice()
        dice.roll()
        print(f"Dice 1: {dice.dice1}")
        print(f"Dice 2: {dice.dice2}")
    elif choice == "n":
        playing = False
        print("Okay, maybe next time!")
    else:
        playing = False
        print("Invalid input. Please enter 'y' or 'n'.")
