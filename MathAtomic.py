from math import inf
from random import randint, uniform


class IntervalIterator:
    def __init__(self, _interval, resolution=1, **kwargs):
        self.resolution = resolution
        self._interval = _interval
        self.start = _interval.lower_bound
        self.curr_point = self.start

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_point >= self._interval.upper_bound:
            raise StopIteration

        self.curr_point += self.resolution
        return self.curr_point


class RandomIterator:
    def __init__(self, _interval, num_iters, **kwargs):
        self._interval = _interval
        self.max_iters = num_iters
        self.curr_iter = 0

    def __iter__(self):
        raise NotImplementedError()


class IntegerRandomIterator(RandomIterator):
    def __iter__(self):
        for _ in range(self.max_iters):
            yield randint(self._interval.lower_bound, self._interval.upper_bound)


class FloatRandomIterator(RandomIterator):
    def __iter__(self):
        for _ in range(self.max_iters):
            yield uniform(self._interval.lower_bound, self._interval.upper_bound)


class FunctionIterator:
    def __init__(self, iterable, _func=lambda x: x):
        self.iterable = iterable
        self._func = _func

    def __iter__(self):
        for x in self.iterable:
            yield self._func(x)


class Interval:
    def __init__(self, **kwargs):
        pass

    def contained_by(self, _rhs):
        pass


class IntervalObject(Interval):
    def __init__(self, lower_bound=0, lower_bound_inclusive=True, upper_bound=0, upper_bound_inclusive=True):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lower_bound_inclusive = lower_bound_inclusive
        self.upper_bound_inclusive = upper_bound_inclusive

    def contained_by(self, _rhs):
        if self.lower_bound < _rhs.lower_bound or self.upper_bound > _rhs.upper_bound:
            return False

        lower_is_within = self.lower_bound > _rhs.lower_bound
        upper_is_within = self.upper_bound < _rhs.upper_bound

        if lower_is_within and upper_is_within:
            return True

        # If the lower bound is not strictly within, it is either equal or outside. If it is not equal, it is outside,
        # which makes lower_is_within False. If it is equal,
        lower_is_within = lower_is_within or (self.lower_bound == _rhs.lower_bound and self.lower_bound_inclusive
                                              and _rhs.lower_bound_inclusive)
        upper_is_within = upper_is_within or (self.upper_bound == _rhs.upper_bound and self.upper_bound_inclusive
                                              and _rhs.upper_bound_inclusive)

        return lower_is_within and upper_is_within

    def is_number(self):
        # Not sure what an open interval on a number means - eg. [0, 0]?
        return self.lower_bound == self.upper_bound

    def __str__(self):
        if self.is_number():
            return "%d" % self.lower_bound
        seperator = ","
        # The isnumber check avoids accidentally
        if isinstance(self.lower_bound, int) and isinstance(self.upper_bound, int):
            seperator = ".."
            if not self.upper_bound_inclusive:
                self.upper_bound -= 1
                self.upper_bound_inclusive = True
            if not self.lower_bound_inclusive:
                self.lower_bound += 1
                self.lower_bound_inclusive = True

        _lhs = "[" if self.lower_bound_inclusive else "("
        _rhs = "]" if self.upper_bound_inclusive else ")"
        return f"{_lhs}{self.lower_bound}{seperator} {self.upper_bound}{_rhs}"

    def copy_from(self, source):
        self.lower_bound = source.lower_bound
        self.upper_bound = source.upper_bound
        self.lower_bound_inclusive = source.lower_bound_inclusive
        self.upper_bound_inclusive = source.upper_bound_inclusive

    def try_merge(self, _rhs):
        """ Returns None if the ranges have no overlap at all, otherwise returns the merged range.

        :param _rhs: The range object with which to merge.
        :return:
        """
        # Returns a single range object.
        # Simple cases: one is a subset of the other.
        if self.contained_by(_rhs):
            return _rhs
        elif _rhs.contained_by(self):
            return self
        # Can't possibly contain the other.
        elif _rhs.lower_bound > self.upper_bound:
            return None
        elif self.lower_bound > _rhs.upper_bound:
            return None
        elif self.upper_bound > _rhs.lower_bound > self.lower_bound:
            return IntervalObject(self.lower_bound, self.lower_bound_inclusive, _rhs.upper_bound,
                                  _rhs.upper_bound_inclusive)
        elif self.upper_bound >= _rhs.lower_bound > self.lower_bound and (
            self.upper_bound_inclusive or _rhs.lower_bound_inclusive):
            return IntervalObject(self.lower_bound, self.lower_bound_inclusive, _rhs.upper_bound,
                                  _rhs.upper_bound_inclusive)
        elif _rhs.upper_bound > self.lower_bound > _rhs.lower_bound:
            return IntervalObject(_rhs.lower_bound, _rhs.lower_bound_inclusive, self.upper_bound,
                                  self.upper_bound_inclusive)
        elif _rhs.upper_bound >= self.lower_bound > _rhs.lower_bound and (
            self.lower_bound_inclusive or _rhs.upper_bound_inclusive):
            return IntervalObject(_rhs.lower_bound, _rhs.lower_bound_inclusive, self.upper_bound,
                                  self.upper_bound_inclusive)
        print("Did I cover all the cases?")
        return None

    def try_merge_rhs_ge(self, _rhs):
        # A version where _rhs is greater than or equal to this range (eg. lower bound is greater or equal).
        # Will always return None where try_merge returns none, but will not always return a valid solution.
        if self.upper_bound > _rhs.lower_bound > self.lower_bound:
            return IntervalObject(self.lower_bound, self.lower_bound_inclusive, _rhs.upper_bound,
                                  _rhs.upper_bound_inclusive)
        elif self.upper_bound >= _rhs.lower_bound > self.lower_bound and (
            self.upper_bound_inclusive or _rhs.lower_bound_inclusive):
            return IntervalObject(self.lower_bound, self.lower_bound_inclusive, _rhs.upper_bound,
                                  _rhs.upper_bound_inclusive)
        return None


class CombinedIntervalObject(Interval):
    def __init__(self, range_objects):
        super().__init__()
        if not isinstance(range_objects, list):
            self.range_objects = [range_objects]
        else:
            self.range_objects = range_objects

    def add_range_object_and_merge(self, range_object):
        self.sort()
        ro_len = len(self.range_objects)
        i = 0
        while i < ro_len:
            ro_i = self.range_objects[i]
            if ro_i.upper_bound < range_object.lower_bound or ro_i.lower_bound > range_object.upper_bound:
                i += 1
            else:
                attempt = ro_i.try_merge(range_object)
                # We've sucessfully found the first index that should be matched - try to match this will all future
                # indices as well, until no more can be found.
                if attempt is not None:
                    # We should try to merge with everything past i as well.
                    j = i + 1
                    self.range_objects[i] = attempt
                    while j < ro_len:
                        attempt = self.range_objects[j].try_merge(range_object)
                        if attempt is not None:
                            self.range_objects.pop(j)
                            self.range_objects[i] = attempt
                            ro_len -= 1
                        else:
                            j = ro_len
                    i = ro_len

                else:
                    i += 1

    def add_range_object(self, range_object):
        self.range_objects.append(range_object)

    def sort(self):
        self.range_objects.sort(key=lambda x: x.lower_bound)

    def merge_ranges(self):
        # Idea to sort the list inspired by https://leetcode.com/problems/merge-intervals/solution/
        self.sort()
        ro_len = len(self.range_objects)
        i = 0
        # Bounded by O(n): if all merges fail, then we have only tried n - 1 merges.
        # If all merges succeed, then we will have done n - 1 merges.
        # This of course assumes that array accesses takes O(1) time.
        while i < ro_len:
            j = i + 1
            # At the start of each loop, the list is sorted. We can do at most ro_len - j - 1 iterations.
            while j < ro_len:
                # Attempt a merge, which takes O(1) time. If successful, move to the next item and reduce the problem
                # space by 1.
                attempt = self.range_objects[i].try_merge_rhs_ge(self.range_objects[j])
                if attempt is None:
                    # exit loop
                    j = ro_len
                else:
                    # Could also pop j in place and continue onwards.
                    # Haven't compared the two options.
                    self.range_objects.pop(j)
                    self.range_objects[i] = attempt
                    ro_len -= 1
            i += 1

    def contained_by(self, _rhs):
        pass

    def is_number(self):
        pass

    def __str__(self):
        return ", ".join(ro.__str__() for ro in self.range_objects)


class MathAtomic:
    def __init__(self, **kwargs):
        self.range_object = None

    def contained_by(self, _rhs):
        return self.range_object.contained_by(_rhs)


class ErrorableAtomic(MathAtomic):
    """ Should possess an error flag indicating whether the input intervals could produce a type error
    (eg. sqrt([-1... 10]), or even the type of error(s) that could be produced. """

    def __init__(self, **kwargs):
        super().__init__()
        self.error = None


class SingleArgumentFunctionAtomic(ErrorableAtomic):
    """ May need to be overriden if the function can produce values outside of the range _func(min) and _func(max) ->
    eg. -x^2, or a sinusoidal function. """

    def __init__(self, _func, _input_interval, test_iterator=None):
        """ _func is the function as a pointer / object. _input is the input interval, and test_iterator is an iterator
        that produces values we can test the function on. If none, we will assume our first answer is correct. """
        # Need to specify every arg that goes into the function.
        super().__init__()
        self._func = _func
        try:
            out_lower = _func(_input_interval.lower_bound)
            out_upper = _func(_input_interval.upper_bound)
            min_arg = min(out_lower, out_upper)
            max_arg = max(out_lower, out_upper)
            self.range_object = IntervalObject(min_arg, _input_interval.lower_bound_inclusive,
                                               max_arg, _input_interval.upper_bound_inclusive)
        except (ValueError, ArithmeticError, ZeroDivisionError) as e:
            self.error = e
            self.range_object = IntervalObject(-inf, True, inf, True)

        if test_iterator is not None:
            if not hasattr(test_iterator, "_interval"):
                raise ValueError("Test iterator does not have a _interval attribute.")
            if test_iterator._interval is not _input_interval:
                raise ValueError(f"Intervals are not the same: {test_iterator._interval} != {_input_interval}")
            outputs = []
            for val in test_iterator:
                try:
                    outputs.append(_func(val))
                except (ValueError, ArithmeticError, ZeroDivisionError) as e:
                    self.error = e
                    self.range_object = IntervalObject(-inf, True, inf, True)
            if self.error is None:
                self.range_object = IntervalObject(min(outputs), True, max(outputs), True)


class AddAtomic(MathAtomic):
    def __init__(self, _lhs=None, _rhs=None):
        super().__init__()
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        self.range_object = IntervalObject(
            _lhs_range.lower_bound + _rhs_range.lower_bound,
            _lhs_range.lower_bound_inclusive and _rhs_range.lower_bound_inclusive,
            _lhs_range.upper_bound + _rhs_range.upper_bound,
            _lhs_range.upper_bound_inclusive and _rhs_range.upper_bound_inclusive
        )
        self._lhs_range = _lhs_range
        self._rhs_range = _rhs_range

    def __str__(self):
        return "%s + %s = %s" % (self._lhs_range.__str__(), self._rhs_range.__str__(), self.range_object.__str__())


class SubtractAtomic(MathAtomic):
    def __init__(self, _lhs=None, _rhs=None):
        super().__init__()
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        self.range_object = IntervalObject(
            _lhs_range.lower_bound - _rhs_range.upper_bound,
            _lhs_range.lower_bound_inclusive and _rhs_range.upper_bound_inclusive,
            _lhs_range.upper_bound - _rhs_range.lower_bound,
            _lhs_range.upper_bound_inclusive and _rhs_range.lower_bound_inclusive
        )
        self._lhs_range = _lhs_range
        self._rhs_range = _rhs_range

    def __str__(self):
        return "%s - %s = %s" % (self._lhs_range.__str__(), self._rhs_range.__str__(), self.range_object.__str__())


class DivisionAtomic(ErrorableAtomic):
    def __init__(self, _lhs=None, _rhs=None):
        super().__init__()
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        self.error = True
        # If division by 0 is possible, ranges must include infinity.
        # TODO: add check for divisor signs.
        if IntervalObject(0, True, 0, True).contained_by(_rhs_range):
            self.range_object = IntervalObject(-inf, True, inf, True)
        elif IntervalObject(0, False, 0, True).contained_by(_rhs_range):
            self.range_object = IntervalObject(-inf, True, inf, True)
        elif IntervalObject(0, True, 0, False).contained_by(_rhs_range):
            self.range_object = IntervalObject(-inf, True, inf, True)
        elif IntervalObject(0, False, 0, False).contained_by(_rhs_range):
            self.range_object = IntervalObject(-inf, True, inf, True)
        else:
            self.error = False
            self.range_object = IntervalObject(
                _lhs_range.lower_bound / _rhs_range.upper_bound,
                _lhs_range.lower_bound_inclusive and _rhs_range.upper_bound_inclusive,
                _lhs_range.upper_bound / _rhs_range.lower_bound,
                _lhs_range.upper_bound_inclusive and _rhs_range.lower_bound_inclusive
            )

        self._lhs_range = _lhs_range
        self._rhs_range = _rhs_range

    def __str__(self):
        return "%s / %s = %s" % (self._lhs_range.__str__(), self._rhs_range.__str__(), self.range_object.__str__())


class MultiplyAtomic(MathAtomic):
    def __init__(self, _lhs=None, _rhs=None):
        super().__init__()
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        self.range_object = IntervalObject(
            _lhs_range.lower_bound * _rhs_range.lower_bound,
            _lhs_range.lower_bound_inclusive and _rhs_range.lower_bound_inclusive,
            _lhs_range.upper_bound * _rhs_range.upper_bound,
            _lhs_range.upper_bound_inclusive and _rhs_range.upper_bound_inclusive
        )

        self._lhs_range = _lhs_range
        self._rhs_range = _rhs_range

    def __str__(self):
        return "%s * %s = %s" % (self._lhs_range.__str__(), self._rhs_range.__str__(), self.range_object.__str__())


class WeakModuloAtomic(MathAtomic):
    """ Should return a possible range, but not a strict range. May include numbers that will never occur. """
    pass


class StrongModuloAtomic(MathAtomic):
    """ Should return the tightest possible range of values - eg. any number not included in the ranges is guaranteed
    to never occur, and every number in the ranges is guaranteed to occur. """
    pass


class ModuloAtomic(MathAtomic):
    def __init__(self, _lhs=None, _rhs=None):
        super().__init__()
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        if IntervalObject(0, True, 0, True).contained_by(_rhs_range):
            self.range_object = IntervalObject(-inf, True, inf, True)
        elif IntervalObject(0, False, 0, True).contained_by(_rhs_range):
            self.range_object = IntervalObject(-inf, True, inf, True)
        elif IntervalObject(0, True, 0, False).contained_by(_rhs_range):
            self.range_object = IntervalObject(-inf, True, inf, True)
        elif IntervalObject(0, False, 0, False).contained_by(_rhs_range):
            self.range_object = IntervalObject(-inf, True, inf, True)
        else:
            # For floats, produces numbers from 0 to floor(n/2). If b in a % b is negative, transfer negative to a and
            # then flip the set. But, do consider for integers: 5 % [1, 2, 3, 4, 5] produces 0, 1, 2, and so on.
            # However, in the case where a mixing of signs can occur: a % [b...c], a is negative,
            # if b > 0 and c <= a, floor(n/2) holds, but we may skip value(s). If c > a, we can produce any number up
            # to c + a. In the case a % [b...c], where b, c are both negative, and abs(b, c) <= a floor(-n/2) holds.
            # Otherwise, we can produce any number from 0 to a+c, or a + b to a + c if b > 0.
            # How do we find examples such as -2 in 5 % [-1 ... -5]? If we're doing "loose" checking, then missing -2 is
            # fine, but if we're doing tight checks, and want to verify that a described output is completely accurate,
            # then this is not acceptable.

            # Lots of situations:
            # One interval is a number - easy if [a.. b] % c, harder if a % [b.. c]
            # One interval includes negative numbers and the other doesn't
            # Both intervals include positive and negative numbers
            # ...
            print("Modulo Atomic incomplete.")
            # We ignore the case where the bounds are different.
            if _rhs_range.lower_bound != _rhs_range.upper_bound:
                self.range_object = IntervalObject(
                    _lhs_range.lower_bound % _rhs_range.lower_bound,
                    _lhs_range.lower_bound_inclusive and _rhs_range.upper_bound_inclusive,
                    _lhs_range.upper_bound % _rhs_range.upper_bound,
                    _lhs_range.upper_bound_inclusive and _rhs_range.lower_bound_inclusive
                )
            elif _lhs_range.lower_bound // _rhs_range.lower_bound == _lhs_range.upper_bound // _rhs_range.lower_bound:
                self.range_object = IntervalObject(
                    _lhs_range.lower_bound % _rhs_range.upper_bound,
                    _lhs_range.lower_bound_inclusive and _rhs_range.upper_bound_inclusive,
                    _lhs_range.upper_bound % _rhs_range.lower_bound,
                    _lhs_range.upper_bound_inclusive and _rhs_range.lower_bound_inclusive
                )
            elif abs(_lhs_range.upper_bound - _lhs_range.lower_bound) // abs(_rhs_range.lower_bound) >= 1:
                self.range_object = IntervalObject(
                    min(_rhs_range.lower_bound, 0),
                    _lhs_range.lower_bound_inclusive and _rhs_range.upper_bound_inclusive,
                    max(_rhs_range.lower_bound, 0),
                    _lhs_range.upper_bound_inclusive and _rhs_range.lower_bound_inclusive
                )
            elif abs(_lhs_range.upper_bound - _lhs_range.lower_bound) // abs(_rhs_range.lower_bound) < 1:
                _mod_lower = _lhs_range.lower_bound % _rhs_range.lower_bound
                _mod_upper = _lhs_range.upper_bound % _rhs_range.lower_bound
                if _mod_lower > _mod_upper:
                    # Need two ranges:
                    self.range_object = CombinedIntervalObject([
                        IntervalObject(
                            0,
                            True,
                            _mod_upper,
                            True
                        ),
                        IntervalObject(
                            _mod_lower,
                            True,
                            _rhs_range.lower_bound - 1,
                            True
                        )
                    ])
            else:
                print("NOT IMPLEMENTED FOR THESE TYPES")
                print((_lhs_range.upper_bound - _lhs_range.lower_bound) // _rhs_range.lower_bound)
                self.range_object = "ERROR"
            # Consider: [1, 10] % [5] -> get 1, 0
            # Should be [1, 5) for float types, [1, 4] for int types
            # Consider, [4, 6] % 5 -> this should be two range objects.
            if _lhs_range.lower_bound % _rhs_range.lower_bound > _lhs_range.upper_bound % _rhs_range.lower_bound:
                pass

        self._lhs_range = _lhs_range
        self._rhs_range = _rhs_range

    def __str__(self):
        return "%s %% %s = %s" % (self._lhs_range.__str__(), self._rhs_range.__str__(), self.range_object.__str__())


if __name__ == "__main__":
    Atomic1 = IntervalObject(1, True, 2, True)
    Atomic2 = IntervalObject(1.1, True, 1.9, True)
    print(Atomic1.contained_by(Atomic2))
    print(Atomic2.contained_by(Atomic1))
    print(Atomic1.contained_by(Atomic1))
    Atomic3 = IntervalObject(1.0, False, 2, True)
    print(Atomic1.contained_by(Atomic3))
    Atomic4 = AddAtomic(Atomic1, Atomic3)
    print(Atomic4)
    Atomic5 = DivisionAtomic(Atomic1, Atomic3)
    print(Atomic5)
    Atomic6 = MultiplyAtomic(Atomic1, Atomic3)
    print(Atomic6)
    Atomic7 = SubtractAtomic(Atomic1, Atomic3)
    print(Atomic7)
    DivByZeroAtomic = DivisionAtomic(Atomic1, IntervalObject(-1, True, 1, True))
    print(DivByZeroAtomic)
    Atomic8 = ModuloAtomic(Atomic1, IntervalObject(1, False, 1, False))
    print(Atomic8)
    Atomic9 = ModuloAtomic(IntervalObject(1, True, 10, True), IntervalObject(5, False, 5, False))
    print(Atomic9)
    Atomic9 = ModuloAtomic(IntervalObject(1, True, 10, True), IntervalObject(-5, False, -5, False))
    print(Atomic9)
    Atomic10 = ModuloAtomic(IntervalObject(1, True, 10, True), IntervalObject(1, False, 5, False))
    print(Atomic10)
    Atomic11 = AddAtomic(IntervalObject(0, False, 0, False), IntervalObject(0, False, 0, False))
    print(Atomic11)
    Atomic12 = ModuloAtomic(IntervalObject(7, False, 13, False), IntervalObject(12, False, 12, False))
    print(Atomic12)

    Atomic13 = IntervalObject(0, False, 10, False)
    Atomic14 = IntervalObject(-10, False, 0, False)
    print(Atomic13.try_merge(Atomic14))

    Combined = CombinedIntervalObject(
        [IntervalObject(0, True, 5, True), IntervalObject(2, True, 19, True), IntervalObject(16, True, 23, False)]
    )
    Combined.merge_ranges()
    print(Combined)
    Combined.add_range_object(IntervalObject(-8, True, -2, False))
    Combined.add_range_object(IntervalObject(28, True, 37, False))
    Combined.add_range_object_and_merge(IntervalObject(20, True, 29, False))
    print(Combined)