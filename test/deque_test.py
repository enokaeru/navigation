from collections import deque, namedtuple
import numpy as np

test = deque(maxlen=3)


for i in range(10):
    test.append(i)
    print(test, len(test))
    print(test.maxlen)

test = deque(maxlen=3)
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

ex = experience(0, np.array([1]), np.array([1]), np.array([1]), False)

test.append(ex)
test.append(ex)
test.append(ex)

e = test[0]

print(e.reward)

