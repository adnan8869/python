# In computer science, a binary search or half-interval search algorithm finds the position of 
# a target value within a sorted array. The binary search algorithm can be classified as a 
# dichotomies divide-and-conquer search algorithm and executes in logarithmic time.
# eg there are 9 indexes in the list and we want to find the data. There are three resources low, high and mid.
# low = 0, high = 8, mid = (low + high) / 2 = 4. when all the resources are calculated then we compare the data with the mid value.and when all are at the same index then we return the index and that will be the output.

def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0

    while low <= high:
        mid = (high + low) // 2

        # If x is greater, ignore left half
        if arr[mid] < x:
            low = mid + 1

        # If x is smaller, ignore right half
        elif arr[mid] > x:
            high = mid - 1

        # x is present at mid
        else:
            return mid

    # If we reach here, then the element was not present
    return -1


# Test array
arr = [6,12,17,23,38,45,77,84,90]

# Function call
result = binary_search(arr, 45)

if result != -1:
    print("Element is present at index", str(result))
else:
    print("Element is not present in array")