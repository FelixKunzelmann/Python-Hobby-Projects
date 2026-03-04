import random
solution: int
choice: int

solution = random.randint(0, 100)
print(solution)
print("Guess a value by typing a Number between 0 and 100")
unsolved = True
while unsolved:
    choice = int(input("What number are you guessing?"))
    if choice < solution:
        print("To small")
    elif choice == solution:
        print("Yea!!! You guessed correctly!!")
        unsolved = False
    elif choice > solution:
        print("Too large!!")
    else:
        print("Ups! Something went wrong!")
