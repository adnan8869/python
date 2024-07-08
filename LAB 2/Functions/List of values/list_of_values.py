# List of Values
def fun(food):
    for i in food:
        print(i)

# fun(["apple", "banana", "cherry"])
fruits = ["apple", "banana", "cherry"]
fun(fruits)



# Program to find the sum of all numbers stored in a list
def sum_list(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

numbers = [1, 2, 3, 4, 5]
print("Sum of all numbers:", sum_list(numbers))


# Program to find the largest number in a list
def largest_number(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num

print("Largest number:", largest_number([45, 32, 89, 78]))


# Program to reverse a list
def reverse_list(lst):
    return lst[::-1]

lst = ['a', 'b', 'c', 'd', 'e']
print("Reversed list:", reverse_list(lst))


# Program to count the occurrences of an element in a list
def count_occurrences(lst, element):
    return lst.count(element)

lst = [1, 4, 2, 7, 1, 4, 2, 1, 4]
element = 4
print(f"Occurrences of {element}:", count_occurrences(lst, element))



# Program to remove duplicates from a list
def remove_duplicates(lst):
    return list(set(lst))   #list -> set automatically remove duplicate values. Then set -> list

lst = [1, 2, 2, 3, 4, 4, 5]
print("List after removing duplicates:", remove_duplicates(lst))