---
layout: section
---

# Python Basics

---
layout: image-right
image: decks/02_python_basics/a-byte-of-python.png
backgroundSize: contain
---

## A Byte of Python

- Written by **Swaroop Chitlur**
- All my slides for teaching Python are adapted from this book!
- Licensed under a **Creative Commons License**
- [<small>Creative Commons Attribution-Noncommercial-Share Alike 3.0 Unported License</small>](https://creativecommons.org/licenses/by-sa/4.0/)
- https://python.swaroopch.com/

---

## Comments

- Use `#` for comments
- They can be above:
  ```python
  # Note that print is a function 
  print('hello world') 
  ```
- Or at the end
  ```python
  print('hello world')  # Note that print is a function
  ```

---

## Literal Constants

- Numbers: `5`, `1.23`
- Strings: `'This is a string'`, `"It's a string!"`
- Boolean: `True`, `False`
- Nothing!: `None`

---

## Numbers
Numbers are mainly of two types - **integers** and **floats**.

### Integers
- An example of an integer is `2` which is just a whole number.

### Floats
- Examples of floating point numbers ("floats") are `3.23` and `52.3e-4`.
- The `e` notation indicates powers of 10. `52.3e-4` => `52.3 * 10^-4^`.

---

## Strings

- Strings are used to represent text.
- You can use single quotes, double quotes, or triple quotes for multi-line strings.
  ```python
  'Hello'
  "Hello"
  '''
  This is a
  multi-line string
  '''
  ```
- Strings are **immutable**

---

## String Interpolation

Python has a powerful string interpolation capability.

```python {monaco-run} {autorun: false}
name = 'Mr. P'
age = 99
print(f'{name} was {age} years old')
```

We can format values while interpolating.

```python {monaco-run} {autorun: false}
weight = 123.456
print(f'Weight is {weight:.2f}')
```

---

## Variables Rules

- The first character of the identifier must be a letter of the alphabet or an underscore (_).
- The rest of the identifier name can consist of letters (uppercase ASCII or lowercase ASCII or Unicode character), underscores (_) or digits (0-9).
- Identifier names are case-sensitive. For example, `myname` and `myName` are not the same.

### Examples

- `age`, `name`, `foo`, `bar`
- `age2`, `name_2`, `foo_bar`
- `_age`, `_name` - starting with `_` denotes "private"
- `AGE`, `NAME` - ALL_CAPS denotes "constant"
- `i`, `j`, `k` - common for iterators in loops (more on this later)
- `df`, `x`, `y`, `X`, `Y` - bad variables but common in data science

---

## Variables Contain Both Data and Types

You can find our the type by using the `type` function.

```python {monaco-run} {autorun: false}
i = 5
print(i)
print(type(i))
```

```python {monaco-run} {autorun: false}
i = 'foo'
print(i)
print(type(i))
```

---

## Logical Operators

```python
True and False
## False

True or False 
## True

not True      
## False
```


---
layout: two-cols-header-2
---

## `if` Statement

The `if` statement is used to conditionally execute a block of code.

### Syntax
```python
if condition:
    statement
```

::left::

```python {monaco-run} {autorun: false}
x = 10
if x > 5:
    print("x is greater than 5")
```

::right::

```python {monaco-run} {autorun: false}
x = 10
y = 3
if x > 5 and y < 7:
    print("x is greater than 5")
```

---

## else Statement

The `else` statement is used to execute a block of code if the condition in the `if` statement is false.

### Syntax

```python
if condition:
    statement
else:
    statement
```

```python {monaco-run} {autorun: false}
x = 10
if x > 15:
    print("x is greater than 15")
else:
    print("x is not greater than 15")
```


---

## `elif` Statement

The `elif` (short for else if) statement is used to check multiple expressions for true and execute a block of code as soon as one of the conditions is true.

### Syntax
```py
if condition1:
    statement
elif condition2:
    statement
else:
    statement
```

```py {monaco-run}
x = 10
if x > 15:
    print("x is greater than 15")
elif x > 5:
    print("x is greater than 5 but not greater than 15")
else:
    print("x is not greater than 5")
```

---

## Loops

Python has two primitive loop commands, `for` and `while`.

### `while` Loops
```python {monaco-run} {autorun: false}
i = 1
while i < 3:
    print(i)
    i += 1
```


### `for` Loops
```python {monaco-run} {autorun: false}
for i in range(1, 3):
    print(i)
```

---
layout: section
hideInToc: true
---

# Python Functions


---

## Python Functions

*In Python, functions are defined with the `def` keyword and use **indentation** (rather than braces) to indicate code blocks. Python also allows **dynamic typing**, meaning you don't have to specify variable types as you might in Java or JavaScript.*

- Use `def` followed by the function name and parentheses.
- The body is indented (traditionally 4 spaces).

```python {monaco-run} {autorun: false}
def say_hello():
    print('Hello, world!')

say_hello()
```

---

## Function Arguments

*Python functions can accept arguments just like Java or JS, but you don't have to specify types by default (you can optionally use type hints).*

```python {monaco-run} {autorun: false}
def print_max(a, b):
    if a > b:
        print(a, 'is maximum')
    else:
        print(b, 'is maximum')

print_max(3, 4)
```

---

## The return Statement

*Unlike Java’s `void` or JS functions that might return `undefined`, Python will return `None` if you don’t explicitly return something.*

```python {monaco-run} {autorun: false}
def maximum(x, y):
    if x > y:
        return x
    else:
        return y

val = maximum(2, 3)
print(val)
```

---

## Keyword Arguments

*Python allows calling functions using **keyword arguments**, so order doesn’t matter.*

```python {monaco-run} {autorun: false}
def print_max(a, b):
    if a > b:
        print(a, 'is maximum')
    else:
        print(b, 'is maximum')

print_max(b=3, a=4)
```

---

## Default Arguments

*Default arguments let you give a function parameter a default value if none is provided.*

```python {monaco-run} {autorun: false}
def say(message, times=1):
    print(message * times)

say('Hello')
say('World', 5)
```

---

## Local Variables

*In Python, variables defined inside a function are local unless otherwise declared (e.g. with `global`).*

```python {monaco-run} {autorun: false}
def func(x):
    print('x is', x)
    x = 2
    print('Changed local x to', x)

x = 50
func(x)
print('x is still', x)
```

---

## Recursion

*Python supports recursion (though the default recursion depth is limited).*

```python {monaco-run} {autorun: false}
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))
```

---

## Docstrings

*Docstrings are how Python does multi-line documentation inside functions (or classes, modules, etc.).*

```python {monaco-run} {autorun: false}
def print_max(a, b):
    """
    Prints the maximum of two numbers.
    """
    a = int(a)
    b = int(b)
    if a > b:
        print(a, 'is maximum')
    else:
        print(b, 'is maximum')

print(print_max.__doc__)
```

---

## Type Hints

*Python’s type hints are optional and aren’t enforced at runtime, but they’re useful for tools (like linters or IDEs).*

```python {monaco-run} {autorun: false}
def print_max(a: int, b: int):
    """Prints the maximum of two numbers."""
    if a > b:
        print(a, 'is maximum')
    else:
        print(b, 'is maximum')

print_max(3, 4)
```

---

## Input/Output Functions

*Python has built-in functions for reading from and writing to the console.*

### Input
Input pauses the program and waits for the user to type something. Whatever the user types is
returned as a string.

```python
text = input('Enter something: ')
```

### Print
Print writes to the console.

```python
print('You entered:', text)
```

---

## Exercise: `is_prime`

*Let's write a function that determines if a number is prime.*

```python {monaco-run} {autorun: false}
def is_prime(n):
    return False

val = is_prime(7)
print(f"7 is {val}")

val = is_prime(14)
print(f"14 is {val}")
```