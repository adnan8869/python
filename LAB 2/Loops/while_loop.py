count = 0
while(count < 5):
    count += 1
    print("Hello World")
else:
    print("Loop finished")


m = 0
while m < 5:
    m += 1
    if m == 3:
        continue
    print("Loop iteration with continue", m)

