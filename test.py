for x in range(100):
    if (x % 3 == 0) and (x % 5 == 0):
        print("Fizzbuzz")
    elif (x % 3 == 0):
        print("Buzz")
    elif (x % 5 == 0):
        print("Fizz")
    else:
        print(x)
