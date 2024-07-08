
# Default Parameter Value


def my_function1(country = "Norway"): 
    print("I am from " + country) 

my_function1("Pakistan")
my_function1("Deutschland")
my_function1()   


# Default parameter in a greeting function
def greet(name, msg="Good morning!"):
    print(f"Hello {name}, {msg}")   #formatted String literals

greet("Adnan")
greet("Zubair", "How do you do?")

# Default parameter for a function that calculates power
def power(base, exponent=2):
    return base ** exponent

print(power(3))
print(power(3, 3))

# Default parameters for a function that formats information
def format_info(name, age=23, city="Berlin"):
    print(f"Name: {name}, Age: {age}, City: {city}")

format_info("Alice", 24, "Los Angeles")
format_info("India", city="Delhi")
format_info("Deutschland")
