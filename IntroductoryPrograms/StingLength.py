string = "Das ist ein Teststring"
lenth = len(string)
print("Die Länge des Strings ist:", lenth)
print(string[-1])
print(string[2:5])
print(string[:7])
name = f"Hallo {string[:1].lower()}{string[1:]}!"
print(name)
