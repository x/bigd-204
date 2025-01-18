---
layout: center
---

# Python Basics

---

# A Byte of Python

- Written by **Swaroop Chitlur**
- All my slides for teaching Python are adapted from this book!
- Licensed under a **Creative Commons License**
- [<small>Creative Commons Attribution-Noncommercial-Share Alike 3.0 Unported License</small>](https://creativecommons.org/licenses/by-sa/4.0/)
- https://python.swaroopch.com/

---
layout: center
---

# `shorturl.at/Oxab1`

---

# Comments

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

# Literal Constants

- Numbers: `5`, `1.23`
- Strings: `'This is a string'`, `"It's a string!"`
- Boolean: `True`, `False`
- Nothing!: `None`

---

# Numbers
Numbers are mainly of two types - **integers** and **floats**.

## Integers
- An example of an integer is `2` which is just a whole number.

## Floats
- Examples of floating point numbers ("floats") are `3.23` and `52.3e-4`.
- The `e` notation indicates powers of 10. `52.3e-4` => `52.3 * 10^-4^`.

---

# Strings

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

---

## Strings (continued)

- Strings are **immutable** (more on that later)
- Use the format method:
  ```python
  name = 'Swaroop'
  age = 20
  print(f'{name} was {age} years old')
  ```

---

## Escape Sequences

- Use `\` for escape sequences
- Examples:
  - `\'` for single quote
  - `\"` for double quote
  - `\\` for backslash
  - `\n` for newline
  - `\t` for tab

---

## Escape Sequences
<br>
```python
>>> print("foo")
foo
```
<br>
```python
>>> print("foo\nbar")
foo
bar
```
<br>
```python
>>> print("foo\n\nbar")
foo

bar
```

---

## Variables

- Variables store information
  ```python
  >> i = 5
  >> print(i)
  5
  >> i = i + 1
  >> print(i)
  6
  ```


---

## Variables Rules

-  The first character of the identifier must be a letter of the alphabet or an underscore (_).
-  The rest of the identifier name can consist of letters (uppercase ASCII or lowercase ASCII or Unicode character), underscores (_) or digits (0-9).
- Identifier names are case-sensitive. For example, `myname` and `myName` are not the same.

---

## Variable Name Examples

- `age`, `name`, `foo`, `bar`
- `age2`, `name_2`, `foo_bar`
- `_age`, `_name` - starting with `_` denotes "private"
- `AGE`, `NAME` - ALL_CAPS denotes "constant"
- `i`, `j`, `k` - common for iterators in loops (more on this later)
- `df`, `x`, `y`, `X`, `Y` - bad variables but common in data science

---

## Variables Contain Both Data and Types

```python
>>> i = 5
>>> type(i)
<class 'int'>
>>> i = 'foo'
>>> type(i)
<class 'str'>
```

---

## Logical and Physical Lines

- **Logical line:** single statement
- **Physical line:** what you see in the editor

```python
i = 5
print(i)
```

```python
i = 5; print(i)
```

---

## Indentation

- Whitespace at the beginning of the line
- Determines grouping of statements
- Use four spaces for indentation
  ```python
  if True:
      print('True')
  else:
      print('False')
  ```

---

## Exercise: Variables

1. Assign your name and age to variables.
2. Calculate how many years it will be until you are 100 years old in a new variable.
3. Print your name, your age, and the number of years until you are 100.

<!-- TODO Setup the jupyter notebooks -->