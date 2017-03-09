from __future__ import division

for i in [1, 2, 3, 4, 5]:
    print i
    for j in [1, 2, 3, 4, 5]:
        print j
        print i + j
    print i
print "done looping"

long_winded_computation = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 +
                           13 + 14 + 15 + 16 + 17 + 18 + 19 + 20)
print long_winded_computation

list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print list_of_lists

easier_to_read_list_of_lists = [[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]]
print easier_to_read_list_of_lists

two_plus_three = 2 + \
    3
print two_plus_three

for i in [1, 2, 3, 4, 5]:

    print i

import re

my_regex = re.compile("[0-9]+")
print my_regex.match("123")
print my_regex.match("abc")

import re as regex

my_regex = regex.compile("[0-9]+")
print my_regex.match("123")
print my_regex.match("abc")

import matplotlib.pyplot as plt

print plt

from collections import defaultdict, Counter

lookup = defaultdict(int)
print lookup

my_counter = Counter()
print my_counter

match = 10
print match

from re import *

print match

print division

def double(x):
    return x * 2
print double(1)

def apply_to_one(f):
    return f(1)

my_double = double
x = apply_to_one(my_double)
print x

y = apply_to_one(lambda x: x + 4)
print y

another_double = lambda x: 2 * x
print another_double(1)

def another_double(x): return 2 * x
print another_double(1)

def my_print(message="my default message"):
    print message

my_print("hello")
my_print()

def subtract(a=0, b=0):
    return a - b

print subtract(10, 5)
print subtract(0, 5)
print subtract(b=5)

single_quoted_string = 'data science'
print single_quoted_string

double_quoted_string = "data science"
print double_quoted_string

tab_string = "\t"
print len(tab_string)

not_tab_string = r"\t"
print len(not_tab_string)

multi_line_string = """This is the first line.
and this is the second line
and this is the third line"""
print multi_line_string

try:
    print 0 / 0
except ZeroDivisionError:
    print "cannot divide by zero"

integer_list = [1, 2, 3]
print integer_list

heterogeneous_list = ["string", 0.1, True]
print heterogeneous_list

list_of_lists = [ integer_list, heterogeneous_list, [] ]
print list_of_lists

list_length = len(integer_list)
print list_length

list_sum = sum(integer_list)
print list_sum

x = range(10)
print x

zero = x[0]
print zero

one = x[1]
print one

nine = x[-1]
print nine

eight = x[-2]
print eight

x[0] = -1
print x

first_three = x[:3]
print first_three

three_to_end = x[3:]
print three_to_end

one_to_four = x[1:5]
print one_to_four

last_three = x[-3:]
print last_three

without_first_and_last = x[1:-1]
print without_first_and_last

copy_of_x = x[:]
print copy_of_x

print 1 in [1, 2, 3]
print 0 in [1, 2, 3]

x = [1, 2, 3]
x.extend([4, 5, 6])
print x

x = [1, 2, 3]
y = x + [4, 5, 6]
print x
print y

x.append(0)
print x

y = x[-1]
print y

z = len(x)
print z

x, y = [1, 2]
print x
print y

_, y = [1, 2]
print y

my_list = [1, 2]
print my_list

my_tuple = (1, 2)
print my_tuple

other_tuple = 3, 4
print other_tuple

my_list[1] = 3
print my_list

try:
    my_tuple[1] = 3
except TypeError:
    print "cannot modify a tuple"

def sum_and_product(x, y):
    return (x + y), (x * y)
sp = sum_and_product(2, 3)
print sp

s, p = sum_and_product(5, 10)
print s
print p

x, y = 1, 2
print x
print y

x, y = y, x
print x
print y

empty_dict = {}
print empty_dict

empty_dict2 = dict()
print empty_dict2

grades = { "Joel": 80, "Tim": 95 }
print grades

joels_grade = grades["Joel"]
print joels_grade

try:
    kates_grade = grades["Kate"]
except KeyError:
    print "no grade for Kate!"

joel_has_grade = "Joel" in grades
print joel_has_grade

kate_has_grade = "Kate" in grades
print kate_has_grade

joel_grade = grades.get("Joel", 0)
print joel_grade

kates_grade = grades.get("Kate", 0)
print kates_grade

no_ones_grade = grades.get("No One")
print no_ones_grade

grades["Tim"] = 99
grades["Kate"] = 100
print grades

num_students = len(grades)
print num_students

tweet = {
    "user": "joelgrus",
    "text": "Data Science is Awesome",
    "retweet_count": 100,
    "hashtags": ["#data", "#science", "#datascience", "#awesome", "#yolo"]
}
print tweet

tweet_keys = tweet.keys()
print tweet_keys

tweet_values = tweet.values()
print tweet_values

tweet_items = tweet.items()
print tweet_items

print "user" in tweet_keys
print "user" in tweet
print "joelgrus" in tweet_values

document = "This is a test. This is another test.".split()

word_counts = {}
for word in document:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1
print word_counts

word_counts = {}
for word in document:
    try:
        word_counts[word] += 1
    except KeyError:
        word_counts[word] = 1
print word_counts

word_counts = {}
for word in document:
    previous_count = word_counts.get(word, 0)
    word_counts[word] = previous_count + 1
print word_counts

word_counts = defaultdict(int)
for word in document:
    word_counts[word] += 1
print word_counts

dd_list = defaultdict(list)
print dd_list

dd_list[2].append(1)
print dd_list

dd_dict = defaultdict(dict)
print dd_dict

dd_dict["Joel"]["City"] = "Seattle"
print dd_dict

f = lambda: [0, 0]
print f()

dd_pair = defaultdict(f)
print dd_pair

dd_pair[2][1] = 1
print dd_pair

c = Counter([0, 1, 2, 0])
print c

word_counts = Counter(document)
print word_counts

for word, count in word_counts.most_common(10):
    print word, count

s = set()
print s

s.add(1)
print s

s.add(2)
print s

s.add(2)
print s

x = len(s)
print x

y = 2 in s
print y

z = 3 in s
print z

hundreds_of_other_words = ["x", "y", "z"]
stopwords_list = ["a", "an", "at"] + hundreds_of_other_words + ["yet", "you"]
print "zip" in stopwords_list

stopwords_set = set(stopwords_list)
print "zip" in stopwords_set

item_list = [1, 2, 3, 1, 2, 3]
num_items = len(item_list)
print num_items

item_set = set(item_list)
print item_set

num_distinct_items = len(item_set)
print num_distinct_items

distinct_item_list = list(item_set)
print distinct_item_list

if 1 > 2:
    message = "if only 1 were greater than two..."
elif 1 > 3:
    message = "elif stands for 'else if'"
else:
    message = "when all else fails use else (if you want to)"
print message

x = 1
parity = "even" if x % 2 == 0 else "odd"
print parity

x = 0
while x < 10:
    print x, "is less than 10"
    x += 1

for x in range(10):
    print x, "is less than 10"

for x in range(10):
    if x == 3:
        continue
    if x == 5:
        break
    print x

one_is_less_than_two = 1 < 2
print one_is_less_than_two

true_equals_false = True == False
print true_equals_false

x = None
print x == None
print x is None

if None:
    print "None"

if []:
    print "[]"

if {}:
    print "{}"

if "":
    print "empty string"

if set():
    print "set()"

if 0:
    print "0"

if 0.0:
    print "0.0"

def some_function_that_returns_a_string():
    return "Hello, world!"

s = some_function_that_returns_a_string()
if s:
    first_char = s[0]
else:
    first_char = ""
print first_char

first_char = s and s[0]
print first_char

x = 1
safe_x = x or 0
print safe_x

print all([True, 1, { 3 }])
print all([True, 1, {}])
print any([True, 1, {}])
print all([])
print any([])

x = [4, 1, 2, 3]
y = sorted(x)
print x
print y

x.sort()
print x

x = sorted([-4, 1, -2, 3], key=abs, reverse=True)
print x

wc = sorted(word_counts.items(), key=lambda (word, count): count, reverse=True)
print wc

even_numbers = [x for x in range(5) if x % 2 == 0]
print even_numbers

squares = [x * x for x in range(5)]
print squares

even_squares = [x * x for x in even_numbers]
print even_squares

square_dict = { x: x * x for x in range(5) }
print square_dict

square_set = { x * x for x in [1, -1] }
print square_set

zeros = [0 for _ in even_numbers]
print zeros

pairs = [(x, y)
         for x in range(10)
         for y in range(10)]
print pairs

increasing_pairs = [(x, y)
                    for x in range(10)
                    for y in range(x + 1, 10)]
print increasing_pairs

def lazy_range(n):
    i = 0
    while i < n:
        yield i
        i += 1
print lazy_range(5)

def do_something_with(i):
    print i

for i in lazy_range(10):
    do_something_with(i)

def natural_numbers():
    n = 1
    while True:
        yield  n
        n += 1

for n in natural_numbers():
    print n
    if n == 100:
        break

lazy_evens_below_20 = (i for i in range(20) if i % 2 == 0)
print lazy_evens_below_20

for i in lazy_evens_below_20:
    print i

import random

four_uniform_random = [random.random() for _ in range(4)]
print four_uniform_random

random.seed(10)
print random.random()

random.seed(10)
print random.random()

for _ in range(10):
    print "10:", random.randrange(10)

for _ in range(10):
    print "3, 5:", random.randrange(3, 6)

up_to_ten = range(10)
print up_to_ten

random.shuffle(up_to_ten)
print up_to_ten

my_best_friend = random.choice(["Alice", "Bob", "Charlie"])
print my_best_friend

lottery_numbers = range(60)
print lottery_numbers

winning_numbers = random.sample(lottery_numbers, 6)
print winning_numbers

four_with_replacement = [random.choice(range(10)) for _ in range(4)]
print four_with_replacement

print all([
    not re.match("a", "cat"),
    re.search("a", "cat"),
    not re.search("c", "dog"),
    3 == len(re.split("[ab]", "carbs")),
    "R-D-" == re.sub("[0-9]", "-", "R2D2")
])

class Set:
    def __init__(self, values=None):
        self.dict = {}

        if values is not None:
            for value in values:
                self.add(value)

    def __repr__(self):
        return "Set: " + str(self.dict.keys())

    def add(self, value):
        self.dict[value] = True

    def contains(self, value):
        return value in self.dict

    def remove(self, value):
        del self.dict[value]

s = Set([1, 2, 3])
print s

print s.contains(4)
s.add(4)
print s.contains(4)

print s.contains(3)
s.remove(3)
print s.contains(3)

def exp(base, power):
    return base ** power
print exp(2, 3)

def two_to_the(power):
    return exp(2, power)
print two_to_the
print two_to_the(3)

from functools import partial

two_to_the = partial(exp, 2)
print two_to_the
print two_to_the(3)

square_of = partial(exp, power=2)
print square_of(3)

def double(x):
    return 2 * x

xs = [1, 2, 3, 4]
twice_xs = [double(x) for x in xs]
print twice_xs

twice_xs = map(double, xs)
print twice_xs

list_doubler = partial(map, double)
twice_xs = list_doubler(xs)
print twice_xs

def multiply(x, y): return x * y

products = map(multiply, [1, 2], [4, 5])
print products

def is_even(x):
    return x % 2 == 0

x_evens = [x for x in xs if is_even(x)]
print x_evens

x_evens = filter(is_even, xs)
print x_evens

list_evener = partial(filter, is_even)
x_evens = list_evener(xs)
print x_evens

x_product = reduce(multiply, xs)
print x_product

list_product = partial(reduce, multiply)
x_product = list_product(xs)
print x_product

documents = ["doc1", "doc2", "doc3"]

def do_something(index, document):
    print index
    print document

for i in range(len(documents)):
    document = documents[i]
    do_something(i, document)

i = 0
for document in documents:
    do_something(i, document)
    i += 1

for i, document in enumerate(documents):
    do_something(i, document)

def do_something(index):
    print index

for i in range(len(documents)): do_something(i)
for i, _ in enumerate(documents): do_something(i)

list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]
print zip(list1, list2)

pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)
print letters
print numbers

print zip(('a', 1), ('b', 2), ('c', 3))

def add(a, b): return a + b

print add(1, 2)
# print add([1, 2])
print add(*[1, 2])

def doubler(f):
    def g(x):
        return 2 * f(x)
    return g

def f1(x):
    return x + 1

g = doubler(f1)
print g(3)
print g(-1)

def f2(x, y):
    return x + y

g = doubler(2)
# print g(1, 2)

def magic(*args, **kwargs):
    print "unnamed args:", args
    print "keyword args:", kwargs

magic(1, 2, key="word", key2="word2")

def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = { "z": 3 }
print other_way_magic(*x_y_list, **z_dict)

def doubler_correct(f):
    def g(*args, **kwargs):
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
print g(1, 2)
