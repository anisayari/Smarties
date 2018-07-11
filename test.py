import time

start_time = time.time()
a=[]
for i in range(0,10000000):
    a.append(1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
a=[]
a = [k for k in range(0,10000000)]
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
a=[]
a = [1]*10000000
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
a=[]
while len(a)<10000000:
    a.append(1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
a=[]
while True:
    a.append(1)
    if len(a)==10000000:
        break
print("--- %s seconds ---" % (time.time() - start_time))
