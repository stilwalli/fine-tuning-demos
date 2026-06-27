try:
    age = int(input("whats your age?"))
    print ("age: ", age)
except ValueError:
    print("invalid input")



print ("this will always print inspite of error")
