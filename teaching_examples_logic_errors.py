"""
Teaching Examples for Common Logic Error Patterns in HumanEval
================================================================

These examples demonstrate key concepts that the model frequently misses.
Each example includes:
1. The concept/pattern
2. Common mistake
3. Correct approach
4. Working implementation
"""

# =============================================================================
# PATTERN 1: Loop Generation (not hardcoding)
# Problem: Model hardcodes a fixed-size list instead of using a loop
# =============================================================================

"""
CONCEPT: Generating Lists with Loops
-------------------------------------
When asked to generate n items, use a loop to generate exactly n items.
Don't hardcode a fixed number of items.

COMMON MISTAKE (HumanEval/100 - make_a_pile):
    def make_a_pile(n):
        if n % 2 == 0:
            return [n, n + 2, n + 4]  # ❌ Only 3 items, but n could be 10!
        else:
            return [n, n + 2, n + 4, n + 6]  # ❌ Only 4 items

CORRECT APPROACH:
    Use a loop that runs exactly n times.
"""

def make_a_pile(n):
    """
    Given a positive integer n, you have to make a pile of n levels of stones.
    The first level has n stones.
    The number of stones in the next level is:
        - the next odd number if n is odd.
        - the next even number if n is even.
    Return the number of stones in each level in a list.

    >>> make_a_pile(3)
    [3, 5, 7]
    >>> make_a_pile(5)
    [5, 7, 9, 11, 13]
    """
    result = []
    current = n
    for i in range(n):  # ✅ Loop exactly n times
        result.append(current)
        current += 2  # Increment by 2 to get next odd/even
    return result


# =============================================================================
# PATTERN 2: Multiple Delimiters
# Problem: Model only handles one delimiter when problem mentions multiple
# =============================================================================

"""
CONCEPT: Splitting on Multiple Delimiters
------------------------------------------
When a problem says "separated by commas OR spaces", you need to handle BOTH.
Don't just split on one delimiter.

COMMON MISTAKE (HumanEval/101 - words_string):
    def words_string(s):
        words = s.split(',')  # ❌ Only splits on comma, ignores spaces!
        return words

    # Fails on: "Hi my name"  -> returns ["Hi my name"] instead of ["Hi", "my", "name"]

CORRECT APPROACH:
    Use replace() to normalize delimiters, or use regex, or split multiple times.
"""

def words_string(s):
    """
    You will be given a string of words separated by commas or spaces.
    Your task is to split the string into words and return an array of the words.

    >>> words_string("Hi, my name is John")
    ['Hi', 'my', 'name', 'is', 'John']
    >>> words_string("One two three")
    ['One', 'two', 'three']
    """
    # ✅ Replace commas with spaces, then split on spaces
    s = s.replace(',', ' ')
    words = s.split()  # split() with no argument splits on any whitespace
    return words


# =============================================================================
# PATTERN 3: Token Parsing (not character iteration)
# Problem: Model iterates characters when should parse multi-character tokens
# =============================================================================

"""
CONCEPT: Parse Tokens, Not Characters
--------------------------------------
When dealing with a string that contains multi-character tokens (like "o|" or ".|"),
you must FIRST split the string into tokens, THEN process each token.
Don't iterate character by character.

COMMON MISTAKE (HumanEval/17 - parse_music):
    def parse_music(music_string):
        result = []
        for note in music_string:  # ❌ Iterates CHARACTERS: 'o', ' ', 'o', '|', ...
            if note == 'o':
                result.append(4)
            elif note == '|':
                result.append(2)
        return result

    # Fails because 'o|' should be ONE token (value 2), not 'o' then '|'

CORRECT APPROACH:
    Split the string into tokens first, then map each token to its value.
"""

def parse_music(music_string: str) -> list:
    """
    Input is a string representing musical notes in a special ASCII format.
    Return list of integers corresponding to how many beats each note lasts.

    Legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quarter note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    # ✅ Split into tokens first
    tokens = music_string.split()

    # Create a mapping for each token
    note_map = {
        'o': 4,
        'o|': 2,
        '.|': 1
    }

    # Map each token to its value
    result = [note_map[token] for token in tokens]
    return result


# =============================================================================
# PATTERN 4: Reading Requirements Carefully (digit sum vs count)
# Problem: Model misunderstands what to compute
# =============================================================================

"""
CONCEPT: Digit Sum with Signed Numbers
---------------------------------------
"Sum of digits" means ADD UP the digits, not just count positive numbers.
Pay attention to special handling for negative numbers.

COMMON MISTAKE (HumanEval/108 - count_nums):
    def count_nums(arr):
        count = 0
        for num in arr:
            if num > 0:  # ❌ Counts positive numbers, not digit sums!
                count += 1
        return count

    # Fails because problem asks for "sum of digits > 0", not "positive numbers"
    # Example: 11 has digit sum = 1+1 = 2 (should count)
    #          -11 has digit sum = -1+1 = 0 (should NOT count)

CORRECT APPROACH:
    Calculate digit sum for each number, handle negative numbers specially.
"""

def count_nums(arr):
    """
    Returns the number of elements which has a sum of digits > 0.
    If a number is negative, then its first signed digit will be negative.
    e.g. -123 has signed digits -1, 2, and 3 (sum = 4)

    >>> count_nums([])
    0
    >>> count_nums([-1, 11, -11])
    1
    >>> count_nums([1, 1, 2])
    3
    """
    count = 0
    for num in arr:
        # ✅ Calculate digit sum
        if num == 0:
            digit_sum = 0
        elif num > 0:
            digit_sum = sum(int(d) for d in str(num))
        else:  # negative
            # First digit is negative, rest are positive
            digits_str = str(num)[1:]  # Remove minus sign
            digit_sum = -int(digits_str[0]) + sum(int(d) for d in digits_str[1:])

        if digit_sum > 0:
            count += 1
    return count


# =============================================================================
# PATTERN 5: Rotation/Cyclic Checking
# Problem: Model only checks direct condition, not all rotations
# =============================================================================

"""
CONCEPT: Checking All Rotations
--------------------------------
When a problem involves rotation/shifting, you must check ALL possible rotations,
not just whether the array is already in the desired state.

COMMON MISTAKE (HumanEval/109 - move_one_ball):
    def move_one_ball(arr):
        if not arr:
            return True
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1]:  # ❌ Only checks if already sorted!
                return False
        return True

    # Fails on [3, 4, 5, 1, 2] which CAN be sorted by rotation

CORRECT APPROACH:
    Check if array is sorted after each possible rotation.
"""

def move_one_ball(arr):
    """
    Determine if it's possible to get an array sorted in non-decreasing order
    by performing right shift operations (rotation).

    >>> move_one_ball([3, 4, 5, 1, 2])
    True
    >>> move_one_ball([3, 5, 4, 1, 2])
    False
    """
    if not arr:
        return True

    # ✅ Try all possible rotations
    n = len(arr)
    for rotation in range(n):
        # Create rotated version
        rotated = arr[rotation:] + arr[:rotation]

        # Check if this rotation is sorted
        is_sorted = True
        for i in range(len(rotated) - 1):
            if rotated[i] > rotated[i + 1]:
                is_sorted = False
                break

        if is_sorted:
            return True

    return False


# =============================================================================
# PATTERN 6: Finding Maximum in Range
# Problem: Model gets stuck or uses wrong logic
# =============================================================================

"""
CONCEPT: Finding Maximum Even Number in Range
----------------------------------------------
When finding the largest even number in a range, iterate BACKWARDS from y to x.
This ensures you find the largest match first.

COMMON MISTAKE (HumanEval/102 - choose_num):
    def choose_num(x, y):
        if x > y:
            return -1
        if x % 2 == 0:  # ❌ Returns x if even, but y might have larger even!
            return x
        if y % 2 == 0:
            return y
        # Gets stuck in repetitive logic...

    # Fails on choose_num(12, 15) -> should return 14, not 12

CORRECT APPROACH:
    Iterate from y down to x, return first even number found.
"""

def choose_num(x, y):
    """
    Returns the biggest even integer in range [x, y] inclusive.
    If there's no such number, return -1.

    >>> choose_num(12, 15)
    14
    >>> choose_num(13, 12)
    -1
    """
    if x > y:
        return -1

    # ✅ Iterate backwards from y to x
    for num in range(y, x - 1, -1):
        if num % 2 == 0:
            return num

    return -1  # No even number found


# =============================================================================
# PATTERN 7: Handling Edge Cases First
# Problem: Model doesn't check empty inputs or special cases
# =============================================================================

"""
CONCEPT: Always Handle Edge Cases First
----------------------------------------
Before implementing main logic, handle special cases:
- Empty inputs ([], "", None)
- Single element
- All same elements
- Boundary conditions

TEMPLATE:
    def solve_problem(input_data):
        # ✅ Handle edge cases FIRST
        if not input_data:
            return default_value
        if len(input_data) == 1:
            return handle_single_case(input_data)

        # Now handle general case
        result = process(input_data)
        return result
"""

def find_max(lst):
    """
    Example: Find maximum element in a list.

    >>> find_max([1, 2, 3])
    3
    >>> find_max([])
    None
    >>> find_max([5])
    5
    """
    # ✅ Handle edge cases first
    if not lst:
        return None
    if len(lst) == 1:
        return lst[0]

    # General case
    return max(lst)


# =============================================================================
# KEY TAKEAWAYS
# =============================================================================

"""
SUMMARY OF COMMON MISTAKES:
1. Hardcoding instead of looping
2. Handling only one delimiter when problem says "OR"
3. Iterating characters instead of parsing tokens
4. Misunderstanding what to compute (count vs sum vs max)
5. Checking only direct condition, not all transformations
6. Poor iteration strategy (not going backwards for max)
7. Forgetting edge cases

DEBUGGING CHECKLIST:
✓ Did I read the requirement carefully?
✓ Did I handle ALL cases mentioned (commas OR spaces)?
✓ Did I use a loop when n is variable?
✓ Did I split into tokens before processing?
✓ Did I check all rotations/transformations?
✓ Did I handle empty inputs?
✓ Did I iterate in the optimal direction?
"""

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
