from math import inf


class RangeObject:
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
            return RangeObject(self.lower_bound, self.lower_bound_inclusive, _rhs.upper_bound,
                               _rhs.upper_bound_inclusive)
        elif self.upper_bound >= _rhs.lower_bound > self.lower_bound and (
                self.upper_bound_inclusive or _rhs.lower_bound_inclusive):
            return RangeObject(self.lower_bound, self.lower_bound_inclusive, _rhs.upper_bound,
                               _rhs.upper_bound_inclusive)
        elif _rhs.upper_bound > self.lower_bound > _rhs.lower_bound:
            return RangeObject(_rhs.lower_bound, _rhs.lower_bound_inclusive, self.upper_bound,
                               self.upper_bound_inclusive)
        elif _rhs.upper_bound >= self.lower_bound > _rhs.lower_bound and (
                self.lower_bound_inclusive or _rhs.upper_bound_inclusive):
            return RangeObject(_rhs.lower_bound, _rhs.lower_bound_inclusive, self.upper_bound,
                               self.upper_bound_inclusive)
        print("Did I cover all the cases?")
        return None

    def try_merge_rhs_ge(self, _rhs):
        # A version where _rhs is greater than or equal to this range (eg. lower bound is greater or equal).
        # Will always return None where try_merge returns none, but will not always return a valid solution.
        if self.upper_bound > _rhs.lower_bound > self.lower_bound:
            return RangeObject(self.lower_bound, self.lower_bound_inclusive, _rhs.upper_bound,
                               _rhs.upper_bound_inclusive)
        elif self.upper_bound >= _rhs.lower_bound > self.lower_bound and (
                self.upper_bound_inclusive or _rhs.lower_bound_inclusive):
            return RangeObject(self.lower_bound, self.lower_bound_inclusive, _rhs.upper_bound,
                               _rhs.upper_bound_inclusive)
        return None


class CombinedRangeObject(RangeObject):
    def __init__(self, range_objects):
        if not isinstance(range_objects, list):
            self.range_objects = [range_objects]
        else:
            self.range_objects = range_objects

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
        if self.lower_bound < _rhs.lower_bound or self.upper_bound > _rhs.upper_bound:
            return False

        lower_is_within = self.lower_bound > _rhs.lower_bound
        upper_is_within = self.upper_bound < _rhs.upper_bound

        if lower_is_within and upper_is_within:
            return True

        lower_is_within = lower_is_within or (self.lower_bound == _rhs.lower_bound and self.lower_bound_inclusive
                                              and _rhs.lower_bound_inclusive)
        upper_is_within = upper_is_within or (self.upper_bound == _rhs.upper_bound and self.upper_bound_inclusive
                                              and _rhs.upper_bound_inclusive)

        return lower_is_within and upper_is_within

    def is_number(self):
        return self.lower_bound == self.upper_bound and self.lower_bound_inclusive is self.upper_bound_inclusive is False

    def __str__(self):
        return ", ".join(ro.__str__() for ro in self.range_objects)


class MathAtomic:
    def __init__(self, **kwargs):
        self.range_object = RangeObject()

    def contained_by(self, _rhs):
        return self.range_object.contained_by(_rhs.range_object)


class AddAtomic(MathAtomic):
    def __init__(self, _lhs=None, _rhs=None):
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        self.range_object = RangeObject(
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
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        self.range_object = RangeObject(
            _lhs_range.lower_bound - _rhs_range.upper_bound,
            _lhs_range.lower_bound_inclusive and _rhs_range.upper_bound_inclusive,
            _lhs_range.upper_bound - _rhs_range.lower_bound,
            _lhs_range.upper_bound_inclusive and _rhs_range.lower_bound_inclusive
        )
        self._lhs_range = _lhs_range
        self._rhs_range = _rhs_range

    def __str__(self):
        return "%s - %s = %s" % (self._lhs_range.__str__(), self._rhs_range.__str__(), self.range_object.__str__())


class DivisionAtomic(MathAtomic):
    def __init__(self, _lhs=None, _rhs=None):
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        # If division by 0 is possible, ranges must include infinity.
        # TODO: add check for divisor signs.
        if RangeObject(0, True, 0, True).contained_by(_rhs_range):
            self.range_object = RangeObject(-inf, True, inf, True)
        elif RangeObject(0, False, 0, True).contained_by(_rhs_range):
            self.range_object = RangeObject(-inf, True, inf, True)
        elif RangeObject(0, True, 0, False).contained_by(_rhs_range):
            self.range_object = RangeObject(-inf, True, inf, True)
        elif RangeObject(0, False, 0, False).contained_by(_rhs_range):
            self.range_object = RangeObject(-inf, True, inf, True)
        else:
            self.range_object = RangeObject(
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
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        self.range_object = RangeObject(
            _lhs_range.lower_bound * _rhs_range.lower_bound,
            _lhs_range.lower_bound_inclusive and _rhs_range.lower_bound_inclusive,
            _lhs_range.upper_bound * _rhs_range.upper_bound,
            _lhs_range.upper_bound_inclusive and _rhs_range.upper_bound_inclusive
        )

        self._lhs_range = _lhs_range
        self._rhs_range = _rhs_range

    def __str__(self):
        return "%s * %s = %s" % (self._lhs_range.__str__(), self._rhs_range.__str__(), self.range_object.__str__())


class ModuloAtomic(MathAtomic):
    def __init__(self, _lhs=None, _rhs=None):
        _lhs_range = _lhs
        _rhs_range = _rhs
        if isinstance(_lhs, MathAtomic):
            _lhs_range = _lhs.range_object
        if isinstance(_rhs, MathAtomic):
            _rhs_range = _rhs.range_object

        if RangeObject(0, True, 0, True).contained_by(_rhs_range):
            self.range_object = RangeObject(-inf, True, inf, True)
        elif RangeObject(0, False, 0, True).contained_by(_rhs_range):
            self.range_object = RangeObject(-inf, True, inf, True)
        elif RangeObject(0, True, 0, False).contained_by(_rhs_range):
            self.range_object = RangeObject(-inf, True, inf, True)
        elif RangeObject(0, False, 0, False).contained_by(_rhs_range):
            self.range_object = RangeObject(-inf, True, inf, True)
        else:
            print("Modulo Atomic incomplete.")
            # We ignore the case where the bounds are different.
            if _rhs_range.lower_bound != _rhs_range.upper_bound:
                self.range_object = RangeObject(
                    _lhs_range.lower_bound % _rhs_range.lower_bound,
                    _lhs_range.lower_bound_inclusive and _rhs_range.upper_bound_inclusive,
                    _lhs_range.upper_bound % _rhs_range.upper_bound,
                    _lhs_range.upper_bound_inclusive and _rhs_range.lower_bound_inclusive
                )
            elif _lhs_range.lower_bound // _rhs_range.lower_bound == _lhs_range.upper_bound // _rhs_range.lower_bound:
                self.range_object = RangeObject(
                    _lhs_range.lower_bound % _rhs_range.upper_bound,
                    _lhs_range.lower_bound_inclusive and _rhs_range.upper_bound_inclusive,
                    _lhs_range.upper_bound % _rhs_range.lower_bound,
                    _lhs_range.upper_bound_inclusive and _rhs_range.lower_bound_inclusive
                )
            elif abs(_lhs_range.upper_bound - _lhs_range.lower_bound) // abs(_rhs_range.lower_bound) >= 1:
                self.range_object = RangeObject(
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
                    self.range_object = CombinedRangeObject([
                        RangeObject(
                            0,
                            True,
                            _mod_upper,
                            True
                        ),
                        RangeObject(
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
    Atomic1 = RangeObject(1, True, 2, True)
    Atomic2 = RangeObject(1.1, True, 1.9, True)
    print(Atomic1.contained_by(Atomic2))
    print(Atomic2.contained_by(Atomic1))
    print(Atomic1.contained_by(Atomic1))
    Atomic3 = RangeObject(1.0, False, 2, True)
    print(Atomic1.contained_by(Atomic3))
    Atomic4 = AddAtomic(Atomic1, Atomic3)
    print(Atomic4)
    Atomic5 = DivisionAtomic(Atomic1, Atomic3)
    print(Atomic5)
    Atomic6 = MultiplyAtomic(Atomic1, Atomic3)
    print(Atomic6)
    Atomic7 = SubtractAtomic(Atomic1, Atomic3)
    print(Atomic7)
    DivByZeroAtomic = DivisionAtomic(Atomic1, RangeObject(-1, True, 1, True))
    print(DivByZeroAtomic)
    Atomic8 = ModuloAtomic(Atomic1, RangeObject(1, False, 1, False))
    print(Atomic8)
    Atomic9 = ModuloAtomic(RangeObject(1, True, 10, True), RangeObject(5, False, 5, False))
    print(Atomic9)
    Atomic9 = ModuloAtomic(RangeObject(1, True, 10, True), RangeObject(-5, False, -5, False))
    print(Atomic9)
    Atomic10 = ModuloAtomic(RangeObject(1, True, 10, True), RangeObject(1, False, 5, False))
    print(Atomic10)
    Atomic11 = AddAtomic(RangeObject(0, False, 0, False), RangeObject(0, False, 0, False))
    print(Atomic11)
    Atomic12 = ModuloAtomic(RangeObject(7, False, 13, False), RangeObject(12, False, 12, False))
    print(Atomic12)

    Atomic13 = RangeObject(0, False, 10, False)
    Atomic14 = RangeObject(-10, False, 0, False)
    print(Atomic13.try_merge(Atomic14))

    Combined = CombinedRangeObject(
        [RangeObject(0, True, 5, True), RangeObject(2, True, 19, True), RangeObject(16, True, 23, False)]
    )
    Combined.merge_ranges()
    print(Combined)
