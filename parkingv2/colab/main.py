from typing import AnyStr, List
import numpy as np
import math
import tensorflow as tf
from functools import wraps, partial
import copy
import os
from enum import Enum
import scipy
import random
import pickle
# import pygame
import time
import sys
import getopt


# all imports


# geom/utils.py

def froze_class(cls):
    """
    function to frozen attributes of class
    you can`t add any more attributes after creation of attributes at the class __init__ function
    idea from https://stackoverflow.com/questions/3603502/prevent-creating-new-attributes-outside-init
    :param cls:
    :return:
    """
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and not hasattr(self, key):
            print("Class {} is frozen. Cannot set {} = {}"
                  .format(cls.__name__, key, value))
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


class SingletonDecorator:
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance


def counting_sort_indexes(tlist_len, indexes_src, k, get_sortkey):
    """ Counting sort algo.
        Args:
            tlist_len: target list to sort
            indexes_src: sorce indexies
            k: max value assume known before hand
            get_sortkey: function to retrieve the key that is apply to elements of tlist
            to be used in the count list index.
            map info to index of the count list.
        Adv:
            The count (after cum sum) will hold the actual position of the element in sorted order
    """

    # Create a count list and using the index to map to the integer in tlist.
    if tlist_len == 0:
        return list()

    count_list = [0] * k

    # iterate the tgt_list to put into count list
    for i in range(tlist_len):
        sortkey = get_sortkey(i)
        assert(sortkey < len(count_list))
        count_list[sortkey] += 1

    # Modify count list such that each index of count list is the combined sum of the previous counts
    # each index indicate the actual position (or sequence) in the output sequence.
    for i in range(1, k):
        count_list[i] += count_list[i-1]

    indexes = [0] * tlist_len
    for i in range(tlist_len-1, -1, -1):
        sortkey = get_sortkey(i)
        indexes[count_list[sortkey]-1] = indexes_src[i]
        count_list[sortkey] -= 1

    return indexes


def counting_sort(tlist, k, get_sortkey):
    """ Counting sort algo.
        Args:
            tlist: target list to sort
            k: max value assume known before hand
            get_sortkey: function to retrieve the key that is apply to elements of tlist
            to be used in the count list index.
            map info to index of the count list.
        Adv:
            The count (after cum sum) will hold the actual position of the element in sorted order
    """

    # Create a count list and using the index to map to the integer in tlist.
    count_list = [0] * k

    # iterate the tgt_list to put into count list
    for item in tlist:
        sortkey = get_sortkey(item)
        assert(sortkey < len(count_list))
        count_list[sortkey] += 1

    # Modify count list such that each index of count list is the combined sum of the previous counts
    # each index indicate the actual position (or sequence) in the output sequence.
    for i in range(1, k):
        count_list[i] += count_list[i-1]

    output = [None] * len(tlist)
    for i in range(len(tlist)-1, -1, -1):
        sortkey = get_sortkey(tlist[i])
        output[count_list[sortkey]-1] = tlist[i]
        count_list[sortkey] -= 1

    return output


def radix_sorted(tlist, k, key):
    def get_sortkey2(item, digit_place=2):
        return (key(item)//10**digit_place) % 10

    result = tlist
    for i in range(k):
        result = counting_sort(result, 10, partial(get_sortkey2, digit_place=i))

    return result


def radix_sorted_indexes(tlist, k, key):
    if not tlist:
        indexes = list()
    else:
        indexes = list(range(len(tlist)))

    def get_sortkey2(idice, digit_place=2):
        index = indexes[idice]
        item = tlist[index]
        k_val = key(item)
        return (k_val // 10 ** digit_place) % 10

    for i in range(k):
        indexes = counting_sort_indexes(len(tlist) if tlist else 0, indexes, 10, partial(get_sortkey2, digit_place=i))

    return indexes



# geom/primitives.py

def between(v1, v2, c):
    v1 = np.float32(v1)
    v2 = np.float32(v2)
    c = np.float32(c)
    c = max((np.float32(0.0), min((np.float32(1.0), c))))
    return v1*(np.float32(1.0) - c) + v2*c


@froze_class
class Point:
    """
    class for point
    you can add its, sub, multiple by value
    """

    def __init__(self, x, y):
        self.x = np.float32(x)
        self.y = np.float32(y)

    def to_numpy_array(self):
        return np.array([self.x, self.y], dtype=np.float32)

    def __eq__(self, other):
        return np.allclose(self.to_numpy_array(), other.to_numpy_array(), rtol=1e-6, atol=1e-2)

    def __add__(self, other):
        pt = Point(x=self.x, y=self.y)
        pt.x += np.float32(other.x)
        pt.y += np.float32(other.y)
        return pt

    def __sub__(self, other):
        pt = Point(x=self.x, y=self.y)
        pt.x -= np.float32(other.x)
        pt.y -= np.float32(other.y)
        return pt

    def __mul__(self, other):
        pt = Point(x=self.x, y=self.y)
        c = np.float32(other)
        pt.x *= c
        pt.y *= c
        return pt

    def rotate(self, pt_base, angle):
        pt = Point(self.x, self.y) - pt_base
        length = pt.length()
        a = math.atan2(pt.y, pt.x)
        angle += a
        pt = Point(math.cos(angle)*length, math.sin(angle)*length)
        pt += pt_base
        return pt

    def transpose(self):
        return Point(self.y, self.x)

    @staticmethod
    def vector_from_angle(angle, r=1.0):
        angle = np.float64(angle)
        return Point(math.cos(angle)*r, math.sin(angle)*r)

    def length(self):
        """
        :return: length from point to the (0,0)
        """
        a = np.array([self.x, self.y], dtype=np.float32)
        a = np.square(a)
        a = np.sum(a)
        return np.sqrt(a)

    @staticmethod
    def scalar(p1, p2):
        l1 = p1.length()
        l2 = p2.length()
        if l1 > 0:
            p1 *= np.float32(1.0)/l1
        if l2 > 0:
            p2 *= np.float32(1.0)/l2
        v = p1.x*p2.x + p1.y*p2.y
        v = max((min((v, np.float32(1.0))), np.float32(-1.0)))
        return np.float32(v)

    @staticmethod
    def scalar_angle(p1, p2):
        return np.float32(math.acos(Point.scalar(p1, p2)))

    @staticmethod
    def between(p1, p2, c):
        assert (isinstance(p1, Point))
        assert (isinstance(p2, Point))
        return Point(between(p1.x, p2.x, c), between(p1.y, p2.y, c))

    pass  # class Point


MAX_LINE_PT_DISTANCE = np.float32(100)


class Line:
    def __init__(self, pt0: Point, pt1: Point):
        self._pts = [copy.copy(pt0), copy.copy(pt1)]

    @property
    def pt0(self):
        return self._pts[0]

    @property
    def pt1(self):
        return self._pts[1]

    @property
    def length(self):
        return (self._pts[1] - self._pts[0]).length()

    def get_pt(self, i: int):
        return self._pts[i]

    def transpose(self):
        pts = [pt.transpose() for pt in self._pts]
        return Line(pts[0], pts[1])

    def rotate(self, pt_base: Point, angle: np.float32):
        pts = [pt.rotate(pt_base, angle) for pt in self._pts]
        return Line(pts[0], pts[1])

    def get_vert_dist(self, x, y):
        pts = (self.pt0, self.pt1)
        min_pt = min(pts, key=lambda val: val.x)
        max_pt = max(pts, key=lambda val: val.x)
        if x < min_pt.x:
            return False, (Point(x, y) - min_pt).length()
        elif x > max_pt.x:
            return False, (Point(x, y) - max_pt).length()

        if self.pt0.x == self.pt1.x:
            ys = (self.pt0.y, self.pt1.y)
            if (y >= min(ys)) and (y <= max(ys)):
                return True, np.float32(0.0)
            else:
                return True, np.float32(min((math.fabs(self.pt0.y - y), math.fabs(self.pt1.y - y))))
        else:
            dx = self.pt1.x - self.pt0.x
            dy = self.pt1.y - self.pt0.y
            c = dy/dx
            x0 = x - self.pt0.x
            d = self.pt0.y + c*x0
            return True, d - y

    def point_at_x(self, x):
        dx = self.pt1.x - self.pt0.x
        dy = self.pt1.y - self.pt0.y
        if dx != np.float32(0.0):
            return True, Point(x, self.pt0.y + dy/dx*(x - self.pt0.x))
        else:
            return False, Point(x, self.pt0.y)

    def point_at_y(self, y):
        dx = self.pt1.x - self.pt0.x
        dy = self.pt1.y - self.pt0.y
        if dy != np.float32(0.0):
            return True, Point(self.pt0.x + dx/dy*(y - self.pt0.y), y)
        else:
            return False, Point(self.pt0.x, y)

    def y_inside(self, y):
        if self.pt0.y <= self.pt1.y:
            return (y >= self.pt0.y) and (y <= self.pt1.y)
        else:
            return (y >= self.pt1.y) and (y <= self.pt0.y)

    def x_inside(self, x):
        if self.pt0.x <= self.pt1.x:
            return (x >= self.pt0.x) and (x <= self.pt1.x)
        else:
            return (x >= self.pt1.x) and (x <= self.pt0.x)

    @staticmethod
    def distance_line_line_old(pt_base: Point, angle, l1, l2):
        assert(isinstance(l1, Line))
        assert(isinstance(l2, Line))
        angle = np.float32(angle)
        l = [l1.rotate(pt_base, -angle), l2.rotate(pt_base, -angle)]
        l = [li.transpose() for li in l]
        x = [(l[0].pt0, 0, 0), (l[0].pt1, 0, 1), (l[1].pt0, 1, 0), (l[1].pt1, 1, 1)]
        x = sorted(x, key=lambda val: val[0].x)
        if x[0][1] == x[1][1]:
            return False, (l[x[1][1]].get_pt(x[1][2]) - l[x[2][1]].get_pt(x[2][2])).length()
        else:
            d1 = l[1 if x[1][1] == 0 else 0].get_vert_dist(x[1][0].x, x[1][0].y)
            d2 = l[1 if x[2][1] == 0 else 0].get_vert_dist(x[2][0].x, x[2][0].y)
            if d1[1]*(-1 if x[1][1] == 0 else 1)*d2[1]*(-1 if x[2][1] == 0 else 1) <= np.float32(0.0):
                return True, np.float32(0.0)
            dists = [d1, d2]
            dists = [(b, np.float32(math.fabs(v))) for b, v in dists]
            return min(dists, key=lambda val: val[1])

    @staticmethod
    def distance_line_line(pt_base: Point, angle, l1, l2):
        assert(isinstance(l1, Line))
        assert(isinstance(l2, Line))
        angle = np.float32(angle)
        l1n = l1.rotate(pt_base, angle)
        if np.isclose(l2.pt0.x, l2.pt1.x):
            is_intersect = l1n.x_inside(l2.pt0.x)
            if is_intersect:
                is_intersect, pt_intersect = l1n.point_at_x(l2.pt0.x)
                is_intersect = is_intersect and l2.y_inside(pt_intersect.y)
        elif np.isclose(l2.pt0.y, l2.pt1.y):
            is_intersect = l1n.y_inside(l2.pt0.y)
            if is_intersect:
                is_intersect, pt_intersect = l1n.point_at_y(l2.pt0.y)
                is_intersect = is_intersect and l2.x_inside(pt_intersect.x)
        else:
            return Line.distance_line_line_old(pt_base, angle, l1, l2)
        return is_intersect, 0

    @staticmethod
    def distance_line_pt_old(pt_base: Point, angle, l, pt: Point):
        assert(isinstance(l, Line))
        angle = np.float32(angle)
        l = l.rotate(pt_base, -angle)
        p = pt.rotate(pt_base, -angle)
        l = l.transpose()
        p = p.transpose()
        b, v = l.get_vert_dist(p.x, p.y)
        return b and (v >=0), v

    @staticmethod
    def distance_line_pt(pt_base: Point, angle, l, pt: Point):
        assert(isinstance(l, Line))
        angle = np.float32(angle)
        assert(pt_base is pt)
        l1 = Line(pt, pt + Point.vector_from_angle(angle, MAX_LINE_PT_DISTANCE))
        pt_intersect = pt
        if np.isclose(l.pt0.x, l.pt1.x):
            is_intersect = l1.x_inside(l.pt0.x)
            if is_intersect:
                is_intersect, pt_intersect = l1.point_at_x(l.pt0.x)
                is_intersect = is_intersect and l.y_inside(pt_intersect.y)
        elif np.isclose(l.pt0.y, l.pt1.y):
            is_intersect = l1.y_inside(l.pt0.y)
            if is_intersect:
                is_intersect, pt_intersect = l1.point_at_y(l.pt0.y)
                is_intersect = is_intersect and l.x_inside(pt_intersect.x)
        else:
            return Line.distance_line_pt_old(pt_base, angle, l, pt)
        return is_intersect, (pt_intersect - pt).length() if is_intersect else MAX_LINE_PT_DISTANCE

    pass  # class Line


@froze_class
class Size:
    """
    class to specify size
    you can add, sub or multiple Size by value
    """

    def __init__(self, *, cx = None, cy = None, sz = None):
        assert(((cx is not None) and (cy is not None)) or (sz is not None))
        if (cx is not None) and (cy is not None):
            self.cx = np.float32(cx)
            self.cy = np.float32(cy)
        elif (sz is not None) and isinstance(sz, Size):
            self.cx = sz.cx
            self.cy = sz.cy
        elif (sz is not None) and isinstance(sz, Point):
            self.cx = sz.x
            self.cy = sz.y
        if self.is_empty():
            self.cx = np.float32(0)
            self.cy = np.float32(0)

    def __eq__(self, other):
        return self.cx == np.float32(other.cx)  \
            and self.cy == np.float32(other.cy)

    def __add__(self, other):
        sz = Size(cx=self.cx, cy=self.cy)
        if isinstance(other, Size):
            sz.cx += np.float32(other.cx)
            sz.cy += np.float32(other.cy)
        elif isinstance(other, Point):
            sz.cx += other.cx
            sz.cy += other.cy
        return sz

    def __sub__(self, other):
        sz = Size(cx=self.cx, cy=self.cy)
        if isinstance(other, Size):
            sz.cx -= np.float32(other.cx)
            sz.cy -= np.float32(other.cy)
        elif isinstance(other, Point):
            sz.cx -= other.cx
            sz.cy -= other.cy
        return sz

    def __mul__(self, other):
        sz = Size(cx=self.cx, cy=self.cy)
        c = np.float32(other)
        sz.cx *= c
        sz.cy *= c
        return sz

    def is_empty(self):
        """
        :return: True if Size object is empty
        """
        return (self.cx <= 0) or (self.cy <= 0)

    pass  # class Size


@froze_class
class Rect:
    """
    class to represent rectagle
    you can work with properties of the class
    """

    def __init__(self, *,
            x=None, y=None, w=None, h=None, sz: Size=None,
            top_left: Point=None, bottom_right: Point=None
            ):
        assert(((x is not None) and (y is not None) and (((w is not None) and (h is not None)) or (sz is not None)))
            or ((top_left is not None) and (bottom_right is not None)))
        if (x is not None) and (y is not None) and (((w is not None) and (h is not None)) or (sz is not None)):
            self._tl = Point(x=x, y=y)
            if sz is not None:
                self._sz = sz
            elif (w is not None) and (h is not None):
                self._sz = Size(cx=w, cy=h)
        elif (top_left is not None) and (bottom_right is not None):
            self._tl = top_left
            self._sz = Size(sz=bottom_right-top_left)

    def is_empty(self):
        """
        :return: True if a rect is empty
        """
        return self._sz.is_empty()

    def __eq__(self, other):
        return self.top_left == other.top_left \
            and self.size == other.size

    @property
    def x(self):
        return self._tl.x

    @x.setter
    def x(self, x):
        self._tl.x = np.float32(x)

    @property
    def left(self):
        return self._tl.x

    @left.setter
    def left(self, x):
        self._tl.x = np.float32(x)

    @property
    def right(self):
        return self._tl.x + self._sz.cx

    @right.setter
    def right(self, right):
        self._sz = Size(cx=np.float32(right)-self._tl.x, cy=self._sz.cy)

    @property
    def y(self):
        return self._tl.y

    @y.setter
    def y(self, y):
        self._tl.y = np.float32(y)

    @property
    def top(self):
        return self._tl.y

    @top.setter
    def top(self, y):
        self._tl.y = np.float32(y)

    @property
    def bottom(self):
        return self._tl.y + self._sz.cy

    @bottom.setter
    def bottom(self, bottom):
        self._sz = Size(cx=self._sz.cx, cy=np.float32(bottom)-self._tl.y)

    @property
    def w(self):
        return self._sz.cx

    @w.setter
    def w(self, w):
        self._sz = Size(cx=np.float32(w), cy=self._sz.cy)

    @property
    def h(self):
        return self._sz.cy

    @h.setter
    def h(self, h):
        self._sz = Size(cx=self._sz.cx, cy=np.float32(h))

    @property
    def top_left(self):
        return copy.copy(self._tl)

    @top_left.setter
    def top_left(self, top_left: Point):
        self._tl = copy.copy(top_left)

    @property
    def top_right(self):
        pt = copy.copy(self._tl)
        pt.x += self._sz.cx
        return pt

    @top_right.setter
    def top_right(self, top_right: Point):
        self._tl.y = top_right.y
        self._sz = Size(cx=top_right.x-self._tl.x, cy=self._sz.cy)

    @property
    def bottom_left(self):
        pt = copy.copy(self._tl)
        pt.y += self._sz.cy
        return pt

    @bottom_left.setter
    def bottom_left(self, bottom_left: Point):
        self._tl.x = bottom_left.x
        self._sz = Size(cx=self._sz.cx, cy=bottom_left.y-self._tl.y)

    @property
    def bottom_right(self):
        pt = copy.copy(self._tl)
        pt += Point(self._sz.cx, self._sz.cy)
        return pt

    @bottom_right.setter
    def bottom_right(self, bottom_right: Point):
        sz = bottom_right - self._tl
        self._sz = Size(sz=sz)

    @property
    def size(self):
        return copy.copy(self._sz)

    @size.setter
    def size(self, size: Size):
        self._sz = size

    def union(self, rc):
        """
        :param rc:
        :return: union to current rect with argument (rc) rect and return that union`s rect
        """
        l = min((self.left, rc.left))
        r = max((self.right, rc.right))
        t = min((self.top, rc.top))
        b = max((self.bottom, rc.bottom))
        return Rect(top_left=Point(l, t), bottom_right=Point(r, b))

    def intersect(self, rc):
        """
        :param rc:
        :return: intersect to current rect with argument (rc) rect and return that intersects`s rect
        """
        l = max((self.left, rc.left))
        r = min((self.right, rc.right))
        t = max((self.top, rc.top))
        b = min((self.bottom, rc.bottom))
        return Rect(top_left=Point(l, t), bottom_right=Point(r, b))

    pass  # class Rect


def normal_distribution_density(x, mean, deviation):
    d2 = math.pow(deviation, 2.0)*2
    return np.float32(math.exp(-math.pow(x-mean,2.0)/d2)/math.pow(math.pi*d2, 0.5))


def angle_diff(a1, a2):
    p1 = Point.vector_from_angle(a1, 1.0)
    p2 = Point.vector_from_angle(a2, 1.0)
    v = p1.x*p2.x + p1.y*p2.y
    v = np.clip(v, np.float32(-1.0), np.float32(1.0))
    return np.arccos(v)



# RLParking/settings.py

def kminh_to_mins(val):
    return np.float32(val)*np.float32(1000.0/(60*60))


def acceleration(val, t, c=1.0):
    return np.float32(val)/np.float32(t)*np.float32(c)


def grad_to_rad(a):
    return np.float32(a)*np.float32(math.pi/180)


def rotate_acceleration(a, t, c=1.0):
    return grad_to_rad(a)/np.float32(t)*np.float32(c)


class VehicleConsts:
    TRANSMISSION_ACCELERATION = np.array([
        -acceleration(kminh_to_mins(40), 40, 0.6),
        -acceleration(kminh_to_mins(40), 10, 0.1),
        np.float32(0.0),
        acceleration(kminh_to_mins(40), 10, 0.1),
        acceleration(kminh_to_mins(40), 10, 0.6),
        ], dtype=np.float32)

    TRANSMISSION_SPEED = np.array([
        kminh_to_mins(-7.5),
        kminh_to_mins(-5.0),
        kminh_to_mins(0),
        kminh_to_mins(5.0),
        kminh_to_mins(10.0),
        kminh_to_mins(15.0),
        ], dtype=np.float32)

    SMALL_SPEED_COEF = np.float32(1e-4)

    ROTATE_ACCELERATION = np.array([
        -rotate_acceleration(45.0, 1.0, 1.0),
        -rotate_acceleration(45.0, 1.0, 0.1),
        np.float32(0.0),
        rotate_acceleration(45.0, 1.0, 0.1),
        rotate_acceleration(45.0, 1.0, 1.0),
        ], dtype=np.float32)

    ROTATE_ANGLE_BOUNDS = np.array([
        grad_to_rad(-45.0),
        grad_to_rad(45.0)
    ], dtype=np.float32)

    BOUNDS_RECT = [Point(2, 1), Point(2, -1), Point(-2, -1), Point(-2, 1)]  # points of the
    CENTER_PT = Point(0, 0)
    AXLE_FRONT = [Point(1.5, 1), Point(1.5, -1)]
    AXLE_BACK = [Point(-1.5, 1), Point(-1.5, -1)]
    WHEEL_SIZE = np.float32(0.5)

    VERBOSE = True

    pass  # class VehicleConsts


class Target:
    """class to update correctly"""

    def __init__(self):
        self.TARGET_POINT = Point(6.5, 0)
        self.TARGET_ANGLE = np.float32(math.pi*0.5)
        self.TARGET_POSITION_DEVIATION = np.float32(5.0)  # 6.5
        self.TARGET_ANGLE_DEVIATION = np.float32(math.pi*0.2)  # 0.2
        self.TARGET_POSITION_RB = np.float32(3.0)  # 2.0

        self.TARGET_POINT_REINFORCE_DISTANCE = np.float32(8.0)
        self.TARGET_POINT_REINFORCE_DISTANCE_NEAR = np.float32(0.5)
        self.TARGET_ANGLE_REINFORCE_DISTANCE = np.float32(math.pi*0.5)
        self.TARGET_VELOCITY_REINFORCE_DISTANCE = np.float32(5.0)
        self.TARGET_REINFORCE_BORDER = np.float32(0.9)

        self.TARGET_POINT_MAX_DISTANCE = np.float32(17.0)
        # self.TARGET_CHANGE_LAMBDA = np.float32(3.33333333e-5)
        self.TARGET_CHANGE_LAMBDA = np.float32(3.333333333e-6)

    def update_reinforce_distances(self, target_dist, angle_dist, velocity_dist):
        target_dist = min((self.TARGET_POINT_REINFORCE_DISTANCE, target_dist))
        lambda_val = self.TARGET_CHANGE_LAMBDA
        if self.TARGET_POINT_REINFORCE_DISTANCE <= self.TARGET_POINT_REINFORCE_DISTANCE_NEAR:
            lambda_val = math.pow(lambda_val, 2.0)
        self.TARGET_POINT_REINFORCE_DISTANCE += np.float32(target_dist - self.TARGET_POINT_REINFORCE_DISTANCE)\
                                                * lambda_val
        if self.TARGET_POINT_REINFORCE_DISTANCE <= self.TARGET_POINT_REINFORCE_DISTANCE_NEAR:
            angle_dist = min((self.TARGET_ANGLE_REINFORCE_DISTANCE, angle_dist))
            self.TARGET_ANGLE_REINFORCE_DISTANCE += np.float32(angle_dist - self.TARGET_ANGLE_REINFORCE_DISTANCE)\
                                                    * lambda_val
            velocity_dist = min((self.TARGET_VELOCITY_REINFORCE_DISTANCE, math.fabs(velocity_dist)))
            self.TARGET_VELOCITY_REINFORCE_DISTANCE += np.float32(velocity_dist
                                                                  - self.TARGET_VELOCITY_REINFORCE_DISTANCE)*lambda_val

    def get_reinforce_params(self):
        return (self.TARGET_POINT_REINFORCE_DISTANCE,
                self.TARGET_ANGLE_REINFORCE_DISTANCE,
               self.TARGET_VELOCITY_REINFORCE_DISTANCE)

    def set_reinforce_params(self, params):
        (self.TARGET_POINT_REINFORCE_DISTANCE,
         self.TARGET_ANGLE_REINFORCE_DISTANCE,
         self.TARGET_VELOCITY_REINFORCE_DISTANCE) = params
        print("set reinforce params = ({}, {}, {})"
              .format(self.TARGET_POINT_REINFORCE_DISTANCE,
                      self.TARGET_ANGLE_REINFORCE_DISTANCE,
                      self.TARGET_VELOCITY_REINFORCE_DISTANCE
                      )
              )

    pass  # Target


class PathesFunc:
    def __init__(self, name, func):
        self._name = name
        self._func = func
        pass

    def f(self):
        return self._func(self._name)

    pass  # class Func


class Pathes:
    def __init__(self, version):
        base_path = "/home/klizardin/tmp/parking/{0}/".format(version)
        self._pathes = {
            "tmp": base_path,
            "base_coordinates": os.path.join(base_path, "base_coordinates/") + "coords",
            "train_state": os.path.join(base_path, "train/") + "states",
            "main_model": os.path.join(base_path, "train/") + "main_model.h5",
        }
        # declare functions
        self.get_temp = PathesFunc("tmp", self._get_path).f
        self.is_temp_exist = PathesFunc("tmp", self._is_exist).f
        self.get_base_coordinates = PathesFunc("base_coordinates", self._get_path).f
        self.has_base_coordinates = PathesFunc("base_coordinates", self._is_exist).f
        self.get_train_state = PathesFunc("train_state", self._get_path).f
        self.has_train_state = PathesFunc("train_state", self._is_exist).f
        self.get_main_model = PathesFunc("main_model", self._get_path).f
        self.has_main_model = PathesFunc("main_model", self._is_exist).f

    def _create_path(self, path: AnyStr):
        p, f = os.path.split(path)
        if not os.path.isdir(p):
            os.makedirs(p)

    def _get_path(self, name):
        assert(name in self._pathes)
        path = self._pathes[name]
        self._create_path(path)
        return path

    def _is_exist(self, name):
        assert (name in self._pathes)
        path = self._pathes[name]
        return os.path.isdir(path) or os.path.isfile(path)

    pass  # class Pathes


class ResetWeightCoefs:
    RESET_WEIGHTS_COEF = np.float32(1.0)

    def clear(self):
        self.RESET_WEIGHTS_COEF = np.float32(1.0)

    def set(self):
        self.RESET_WEIGHTS_COEF = np.float32(5e-1)

    pass  # class ResetWeightCoefs


class Settings:
    VEHICLE = VehicleConsts()

    SCREEN_SIZE = Size(cx=50, cy=50)
    STATIC_VEHICLES = [
        (VEHICLE.BOUNDS_RECT, VEHICLE.CENTER_PT, Point(6.5, 5.5), np.float32(math.pi*0.5)),
        (VEHICLE.BOUNDS_RECT, VEHICLE.CENTER_PT, Point(6.5, -5.5), np.float32(math.pi*0.5)),
    ]

    WALLS = [
        Line(Point(8, 15), Point(8, -15)),
        Line(Point(8, 15), Point(-3, 15)),
        Line(Point(-3, 15), Point(-3, -15)),
        Line(Point(-3, -15), Point(8, -15)),
    ]

    TARGET = Target()  # need to correctly update params

    VISUALIZE_START_POSITIONS_WEIGHTS = False
    BASE_POS1_FOR_POS_GENERATOR = Point(-10.0, 15.0)
    BASE_POS2_FOR_POS_GENERATOR = Point(-10.0, -15.0)
    MAX_LENGTH_FOR_POS_GENERATOR = 20
    VERSION = "0.0.1"

    MAX_INPUT_LINE_LENGTH = np.float32(15.0)
    MIN_INPUT_LINE_LENGTH = np.float32(0.999)

    INPUT_SHOW_RECT = Rect(top_left=Point(-20.0, -24.0), bottom_right=Point(20.0, -19.0))
    NET_SHOW_RECT = Rect(top_left=Point(-20.0, 19.0), bottom_right=Point(20.0, 24.0))

    VISUALIZE_INPUT_HISTORY = True
    VEHICLE_STATE_MAX_LENGTH = 16
    VEHICLE_STATE_INDEXES = [0, 3, 15]

    START_POSITIONS = [
        Line(Point(4.5, 12), Point(-1.5, 12)),
        Line(Point(4.5, -12), Point(-1.5, -12)),
    ]

    PATHES = Pathes(VERSION)
    BASE_COORDINATES_COUNT = 10000

    SENSOR_INPUTS_SIZE = 4 + len(VEHICLE_STATE_INDEXES)*3

    ANGLE_FOR_INPUT = 30
    ANGLES_FOR_OPERATIONS = 20
    ANGLE_FOR_OPERATION_MIN = grad_to_rad(-30.0)
    ANGLE_FOR_OPERATION_MAX = grad_to_rad(30.0)
    OPERATIONS_COUNT = ANGLES_FOR_OPERATIONS + 2
    NET_INPUT_SIZE = int((360//ANGLE_FOR_INPUT)*len(VEHICLE_STATE_INDEXES) + SENSOR_INPUTS_SIZE + OPERATIONS_COUNT)
    NET1_FC_SIZE1 = 512
    NET_OPERATION_EXTRA_MIN_SIZE = 128
    NET_OPERATION_ITEM_SIZE = int((NET1_FC_SIZE1 - NET_OPERATION_EXTRA_MIN_SIZE)//OPERATIONS_COUNT)
    NET_OPERATION_EXTRA_SIZE = NET1_FC_SIZE1 - NET_OPERATION_ITEM_SIZE*OPERATIONS_COUNT
    NET1_FC_SIZE2 = 512
    NET1_FC_SIZE3 = 512
    NET_LAYER1_ACTIVATION = tf.nn.leaky_relu
    NET_LAYER2_ACTIVATION = tf.nn.leaky_relu
    NET1_FC_DROPOUT_VALUE1 = 0.2
    NET1_FC_DROPOUT_VALUE2 = 0.2

    VERBOSE_TRAIN_DB = True
    VERBOSE_VEHICLE_COMMANDS = False
    VERBOSE_MAIN_MODEL_SAVE = True

    REWARD_KEY_COLLISION = "collision"
    REWARD_KEY_POS = "pos"
    REWARD_KEY_STAND_POS = "stand_pos"
    REWARD_KEY_STAND_TIME = "stand_time"
    REWARD_KEY_VELOCITY = "velocity"
    REWARD_KEY_ANGLE = "angle"
    REWARD_KEY_TIME = "time"
    REWARD_KEY_DRIVE_TIME = "drive_time"

    REINFORCE_DONE = np.float32(1.0)
    REINFORCE_FAIL = np.float32(-1.0)
    REINFORCE_NONE = np.float32(0.0)

    REWARD_TO_VALUES_COEF = np.float32(0.3333333)
    VALUES_TO_REWARD_COEF = np.float32(1.0/REWARD_TO_VALUES_COEF)
    RL_RO_COEF = np.float32(0.99)
    RL_REWARD_COEF = np.float32(0.99)

    STATES_TO_TRAIN_BATCH_SIZE = 32
    STATES_TO_TRAIN = STATES_TO_TRAIN_BATCH_SIZE*4

    MAIN_MODEL_SAVE_INTERVAL = 60*25
    VALUE_MORE_THAN_MAX_NET_RESULT = 1e2

    RL_SEARCH_COEF = 0.01
    RL_PRETRAIN_SEARCH_COEF = 0.1
    RL_SEARCH_USE_ALPHA_ALGORITHM = False
    USE_LEARNING_BOOST = True

    VEHICLE_UI_BOT_STEPS_PER_SECOND = np.float32(3.0)
    VEHICLE_STEP_DURATION = np.float32(1.0/VEHICLE_UI_BOT_STEPS_PER_SECOND)
    VEHICLE_MAX_DRIVE_DURATION = np.float32(60.0*2.0)
    VEHICLE_MAX_STAND_DURATION = np.float32(5)

    GAME_STATES_IN_DB_SIZE_MAX = 1024*31
    GAME_STATES_IN_DB_SIZE_MAX2 = 1024*32
    GAME_STATES_IN_DB_TO_START_TRAIN = 1024*30
    GAME_STATES_IN_DB_STEP_SIZE = 100
    GAME_STATES_IN_DB_REWARD_RATIO = np.float32(0.3333333)
    GAME_STATE_GETTED_MAX_COUNT = 30

    STATE_LEARN_VALUES_COUNT = 8
    SET_OPERATION_VALUES_PROB = np.float32(0.1)

    RESET_WEIGHTS_COEF = ResetWeightCoefs()

    pass  # class Settings


settings = Settings()

# RLParking/request.py


class RequestType(Enum):
    NOP = 0
    SAVE_TRAIN_STATE = 1
    GET_BEST_OPERATION = 2
    SEARCH_OPERATION = 3
    RL_TRAIN = 4

    pass  # class RequestType


class Request:
    def __init__(self,
        type: RequestType,
        inputs=None,
        ops=None,
        state=None,
        next_state=None,
        reward=None,
        values=None,
        final_state=None,
        ):
        self.type = type
        self.inputs = inputs
        self.ops = ops
        self.state = state
        self.next_state = next_state
        self.reward = reward
        self.ops = ops
        self.final_state = final_state
        self.values = values
        self.results = None
        self._check()
        self.processed = False

    def _check(self):
        assert(self.type is not None)
        if self.is_type(RequestType.GET_BEST_OPERATION) or self.is_type(RequestType.SEARCH_OPERATION):
            assert(self.inputs is not None)
            assert(self.ops is not None)
        elif self.is_type(RequestType.SAVE_TRAIN_STATE):
            assert(self.state is not None)
            assert(self.next_state is not None)
            assert(self.reward is not None)
            assert(self.final_state is not None)
        elif self.is_type(RequestType.RL_TRAIN):
            assert(self.inputs is not None)
            assert(self.ops is not None)
            assert(self.values is not None)

    def is_type(self, type: RequestType):
        return self.type.value == type.value

    def in_types(self, types: List[RequestType]):
        return any([type.value == self.type.value for type in types])

    def get_best_operation(self):
        return np.argmax(self.results)

    def get_best_operation_value(self):
        return np.max(self.results)

    pass  # class Request


# RLParking/reward.py

def calc_fail_value(max_fail_value: np.float32, target_distance, target_max_distance):
    val = max_fail_value*(math.log(target_distance + 1.0)/math.log(target_max_distance + 1.0)*0.9 + 0.1)
    if math.fabs(val) >= math.fabs(max_fail_value):
        return max_fail_value
    else:
        return np.float32(val)


def calc_done_value(max_done_value: np.float32, target_distance, target_max_distance):
    target_distance = min((target_distance, target_max_distance))
    val = max_done_value \
          * (math.log(target_max_distance - target_distance + 1.0)/math.log(target_max_distance + 1.0)*0.9 + 0.1)
    if math.fabs(val) >= math.fabs(max_done_value):
        return max_done_value
    else:
        return np.float32(val)


def get_reward(old_values, new_values):
    assert(settings.REWARD_KEY_COLLISION in new_values)
    assert(settings.REWARD_KEY_DRIVE_TIME in new_values)
    assert((settings.REWARD_KEY_POS in old_values) and (settings.REWARD_KEY_POS in new_values))
    assert((settings.REWARD_KEY_VELOCITY in old_values) and (settings.REWARD_KEY_VELOCITY in new_values))
    assert((settings.REWARD_KEY_ANGLE in old_values) and (settings.REWARD_KEY_ANGLE in new_values))
    assert((settings.REWARD_KEY_TIME in old_values) and (settings.REWARD_KEY_TIME in new_values))

    target_dist = (new_values[settings.REWARD_KEY_POS] - settings.TARGET.TARGET_POINT).length()
    if (new_values[settings.REWARD_KEY_COLLISION]
            or (target_dist > settings.TARGET.TARGET_POINT_MAX_DISTANCE)
            or (new_values[settings.REWARD_KEY_DRIVE_TIME] > settings.VEHICLE_MAX_DRIVE_DURATION)
        ):
        return calc_fail_value(settings.REINFORCE_FAIL, target_dist, settings.TARGET.TARGET_POINT_MAX_DISTANCE), True

    if ((settings.REWARD_KEY_STAND_TIME in old_values)
        and (
            np.fabs(new_values[settings.REWARD_KEY_TIME] - old_values[settings.REWARD_KEY_STAND_TIME])
            > settings.VEHICLE_MAX_STAND_DURATION)
        ):
        return calc_fail_value(settings.REINFORCE_FAIL, target_dist, settings.TARGET.TARGET_POINT_MAX_DISTANCE), True

    angle_dist = angle_diff(new_values[settings.REWARD_KEY_ANGLE], settings.TARGET.TARGET_ANGLE)

    # bound to [0 .. math.pi*0.5]
    angle_dist = np.float32(math.fmod(angle_dist + math.pi, math.pi))
    angle_dist = np.float32(math.fmod(angle_dist + math.pi, math.pi))
    if angle_dist > math.pi*0.5:
        angle_dist = np.float32(math.pi - angle_dist)

    velocity_dist = new_values[settings.REWARD_KEY_VELOCITY]
    td = settings.TARGET.TARGET_POINT_REINFORCE_DISTANCE * settings.TARGET.TARGET_REINFORCE_BORDER
    ta = settings.TARGET.TARGET_ANGLE_REINFORCE_DISTANCE * settings.TARGET.TARGET_REINFORCE_BORDER
    tv = settings.TARGET.TARGET_VELOCITY_REINFORCE_DISTANCE*settings.TARGET.TARGET_REINFORCE_BORDER
    if ((target_dist <= td)
        and (angle_dist <= ta)
        and (np.fabs(velocity_dist) <= tv)
        ):
        settings.TARGET.update_reinforce_distances(target_dist, angle_dist, velocity_dist)
        return calc_done_value(settings.REINFORCE_DONE, target_dist, settings.TARGET.TARGET_POINT_MAX_DISTANCE), True

    return settings.REINFORCE_NONE, False


class Reward:
    def __init__(self):
        self._old_values = dict()
        self._last_id = 0

    def set_values(self, obj_id, values_dict):
        if obj_id not in self._old_values:
            self._old_values[obj_id] = dict()
        for k, v in values_dict.items():
            self._old_values[obj_id][k] = v

    def get_reward(self, obj_id, new_values_dict):
        if obj_id not in self._old_values:
            return settings.REINFORCE_NONE
        stand_pos = None
        stand_time = None
        if ((settings.REWARD_KEY_STAND_POS in self._old_values[obj_id])
                and (settings.REWARD_KEY_STAND_TIME in self._old_values[obj_id])
            ):
            stand_pos = self._old_values[obj_id][settings.REWARD_KEY_STAND_POS]
            stand_time = self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME]
            if not np.isclose((stand_pos - new_values_dict[settings.REWARD_KEY_POS]).length(), np.float32(0.0)):
                stand_pos = None
                stand_time = None

        if stand_pos is not None:
            self._old_values[obj_id][settings.REWARD_KEY_STAND_POS] = stand_pos
            self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME] = stand_time
        else:
            if settings.REWARD_KEY_STAND_POS in self._old_values[obj_id]:
                del self._old_values[obj_id][settings.REWARD_KEY_STAND_POS]
            if settings.REWARD_KEY_STAND_TIME in self._old_values[obj_id]:
                del self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME]

        val = get_reward(self._old_values[obj_id] if obj_id in self._old_values else dict(), new_values_dict)

        if ((settings.REWARD_KEY_STAND_POS in self._old_values[obj_id])
            and (settings.REWARD_KEY_STAND_TIME in self._old_values[obj_id])
            ):
            stand_pos = self._old_values[obj_id][settings.REWARD_KEY_STAND_POS]
            stand_time = self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME]
        elif np.isclose(
                (self._old_values[obj_id][settings.REWARD_KEY_POS]
                 - new_values_dict[settings.REWARD_KEY_POS]).length()
                , np.float32(0.0)
            ):
            stand_pos = self._old_values[obj_id][settings.REWARD_KEY_POS]
            stand_time = self._old_values[obj_id][settings.REWARD_KEY_TIME]
        else:
            stand_pos = None
            stand_time = None

        del self._old_values[obj_id]

        if stand_pos is not None:
            self._old_values[obj_id] = dict()
            self._old_values[obj_id][settings.REWARD_KEY_STAND_POS] = stand_pos
            self._old_values[obj_id][settings.REWARD_KEY_STAND_TIME] = stand_time
        return val

    def get_next_id(self):
        res = self._last_id
        self._last_id += 1
        return res

    pass  # class Reward


Reward = SingletonDecorator(Reward)



# env/vehicle.py

def get_lines(bounds):
    pt0 = bounds[-1]
    lines = list()
    for pt in bounds:
        lines.append(Line(pt0, pt))
        pt0 = pt
    return lines


class StaticVehicle:
    def __init__(self, static_vehicle_pos_info):
        bp = static_vehicle_pos_info[1]
        self._pos = static_vehicle_pos_info[2]
        self._angle = static_vehicle_pos_info[3]
        self._pts = [
            pt.rotate(bp, self._angle) + self._pos
            for pt in static_vehicle_pos_info[0]
            ]

    @property
    def bounds(self):
        return [copy.copy(pt) for pt in self._pts]

    @property
    def position(self):
        return copy.copy(self._pos)

    @property
    def rotation(self):
        return self._angle

    pass  # class StaticVehicle


class StaticEnv:
    def __init__(self):
        self.static_vehicles = [StaticVehicle(svi) for svi in settings.STATIC_VEHICLES]
        self.walls = settings.WALLS
        pass

    def get_lines(self):
        all_lines = list()
        for sv in self.static_vehicles:
            lines = get_lines(sv.bounds)
            all_lines.extend(lines)
        all_lines.extend(self.walls)
        return all_lines

    pass  # class StaticEnv


class VehicleOp(Enum):
    NONE = 0
    ACCELERATE_BACKWARD_2 = 1
    ACCELERATE_BACKWARD_1 = 2
    ACCELERATE_NONE = 3
    ACCELERATE_FORWARD_1 = 4
    ACCELERATE_FORWARD_2 = 5
    ROTATE_RIGHT_2 = 7
    ROTATE_RIGHT_1 = 7
    ROTATE_NONE = 8
    ROTATE_LEFT_1 = 9
    ROTATE_LEFT_2 = 10

    pass  # class VehicleOp


VEHICLE_ACCELERATE_OPS = [
    VehicleOp.ACCELERATE_BACKWARD_2, VehicleOp.ACCELERATE_BACKWARD_1,
    VehicleOp.ACCELERATE_NONE,
    VehicleOp.ACCELERATE_FORWARD_1, VehicleOp.ACCELERATE_FORWARD_2
]


VEHICLE_ROTATE_OPS = [
    VehicleOp.ROTATE_RIGHT_2, VehicleOp.ROTATE_RIGHT_1,
    VehicleOp.ROTATE_NONE,
    VehicleOp.ROTATE_LEFT_1, VehicleOp.ROTATE_LEFT_2
]


# VEHICLE_TRANSMITE_OPS = [VehicleOp.FORWARD, VehicleOp.BACKWARD]


VEHICLE_POSSIBLE_OPERATIONS = [
    (VehicleOp.NONE,),
]

# for o in VEHICLE_TRANSMITE_OPS:
#    VEHICLE_POSSIBLE_OPERATIONS.append((o,))


def init_all_ops(pos_ops):
    for rotate_op in VEHICLE_ROTATE_OPS:
        for angle_op in VEHICLE_ACCELERATE_OPS:
            if ((angle_op.value == VehicleOp.ACCELERATE_NONE.value)
                and (rotate_op.value == VehicleOp.ROTATE_NONE.value)
            ):
                continue
            pos_ops.append((rotate_op, angle_op))
    return pos_ops


VEHICLE_POSSIBLE_OPERATIONS = init_all_ops(VEHICLE_POSSIBLE_OPERATIONS)


def get_breaks_count():
    return int(np.sum(np.where(settings.VEHICLE.TRANSMISSION_ACCELERATION < 0.0, 1, 0)))


def get_accelaration_count():
    return int(np.sum(np.where(settings.VEHICLE.TRANSMISSION_ACCELERATION > 0.0, 1, 0)))


@froze_class
class VehicleOperation:
    def __init__(self, op_id: VehicleOp, name: AnyStr, forward: bool = None,
                 acceleration: int = None, rotate: int = None):
        self.op_id = op_id
        self.name = name
        self.forward = forward
        if acceleration is not None:
            self.break_value = get_breaks_count() - acceleration if acceleration <= get_breaks_count() else None
            self.accelaration_value = acceleration - get_breaks_count() if acceleration > get_breaks_count() else None
        else:
            self.break_value = None
            self.accelaration_value = None
        self.rotate = rotate

    pass  # class VehicleOperation


VEHICLE_SINGLE_OPERATIONS = {
    VehicleOp.NONE:  VehicleOperation(VehicleOp.NONE, "none"),
    # VehicleOp.FORWARD : VehicleOperation(VehicleOp.FORWARD, "forward", forward=True),
    # VehicleOp.BACKWARD : VehicleOperation(VehicleOp.BACKWARD, "backward", forward=False),
    VehicleOp.ACCELERATE_BACKWARD_2: VehicleOperation(VehicleOp.ACCELERATE_BACKWARD_2, "break_2", acceleration=0),
    VehicleOp.ACCELERATE_BACKWARD_1: VehicleOperation(VehicleOp.ACCELERATE_BACKWARD_1, "break_1", acceleration=1),
    VehicleOp.ACCELERATE_NONE: VehicleOperation(VehicleOp.ACCELERATE_NONE, "accelerate_0", acceleration=2),
    VehicleOp.ACCELERATE_FORWARD_1: VehicleOperation(VehicleOp.ACCELERATE_FORWARD_1, "accelerate_1", acceleration=3),
    VehicleOp.ACCELERATE_FORWARD_2: VehicleOperation(VehicleOp.ACCELERATE_FORWARD_2, "accelerate_2", acceleration=4),
    VehicleOp.ROTATE_LEFT_2: VehicleOperation(VehicleOp.ROTATE_LEFT_2, "left_2", rotate=-2),
    VehicleOp.ROTATE_LEFT_1: VehicleOperation(VehicleOp.ROTATE_LEFT_1, "left_1", rotate=-1),
    VehicleOp.ROTATE_NONE: VehicleOperation(VehicleOp.ROTATE_NONE, "rotate_none", rotate=0),
    VehicleOp.ROTATE_RIGHT_1: VehicleOperation(VehicleOp.ROTATE_RIGHT_1, "right_1", rotate=1),
    VehicleOp.ROTATE_RIGHT_2: VehicleOperation(VehicleOp.ROTATE_RIGHT_2, "right_2", rotate=2),
}

# VEHICLE_POSSIBLE_OPERATIONS = [
#    (VEHICLE_SINGLE_OPERATIONS[i] for i in line)
#    for line in VEHICLE_POSSIBLE_OPERATIONS
# ]


@froze_class
class VehiclePos:
    def __init__(self, pos: Point, angle: np.float32, t: np.float32):
        self.pos = pos
        self.angle = angle
        self.t = t

    pass  # class VehiclePos


class VehicleState:
    def __init__(self):
        self.velocity = np.float32(0.0)
        self.acceleration = np.float32(0.0)
        self.front = True
        self.wheel_angle = np.float32(0.0)
        self.wheel_delta_angle = np.float32(0.0)
        self.states = list()
        self.commads = (VehicleOp.NONE,)
        self.last_transmite_op = None
        self.last_rotate_op = None
        self.last_accelerate_op = None
        self.clear()

    def add(self, dt: np.float32):
        p = VehiclePos(copy.deepcopy(self.pos), self.angle, self.last_time + dt)
        if len(self.states) < settings.VEHICLE_STATE_MAX_LENGTH:
            self.states.append(p)
        else:
            self.states[:-1] = self.states[1:]
            self.states[-1] = p

    def replace(self, pos: Point, angle: np.float32, t: np.float32):
        p = VehiclePos(pos, angle, t)
        self.states[-1] = p

    def get_at(self, i):
        i = min((i+1, len(self.states)))
        return self.states[-i]

    @property
    def pos(self):
        return self.states[-1].pos

    @pos.setter
    def pos(self, p: Point):
        self.states[-1].pos = p

    @property
    def angle(self):
        return self.states[-1].angle

    @angle.setter
    def angle(self, angle):
        self.states[-1].angle = np.float32(angle)

    @property
    def last_time(self):
        return self.states[-1].t

    def set_current(self,
            velocity: np.float32, acceleration: np.float32,
            front: bool,
            wheel_angle: np.float32, wheel_delta_angle: np.float32):
        self.velocity = velocity
        self.acceleration = acceleration
        self.front = front
        self.wheel_angle = wheel_angle
        self.wheel_delta_angle = wheel_delta_angle

    def clear(self):
        self.states = list()
        self.velocity = np.float32(0.0)
        self.acceleration = np.float32(0.0)
        self.front = True
        self.wheel_angle = np.float32(0.0)
        self.wheel_delta_angle = np.float32(0.0)
        self.states = list()
        self.states.append(VehiclePos(Point(0, 0), np.float32(0.0), np.float32(0.0)))
        self.commads = (VehicleOp.NONE,)
        self.last_transmite_op = None
        self.last_rotate_op = None
        self.last_accelerate_op = None

    @staticmethod
    def _cmds_of_ops(cmds, ops):
        return [
            op
            for op in cmds
            if any([top.value == op.value for top in ops])
        ]

    def set_commands(self, cmds):
        self.commads = cmds
        # transmite_cmds = VehicleState._cmds_of_ops(cmds, VEHICLE_TRANSMITE_OPS)
        # if transmite_cmds:
        #    self.last_transmite_op = transmite_cmds[0]
        rotate_cmds = VehicleState._cmds_of_ops(cmds, VEHICLE_ROTATE_OPS)
        if rotate_cmds:
            self.last_rotate_op = rotate_cmds[0]
        accelerate_cmds = VehicleState._cmds_of_ops(cmds, VEHICLE_ACCELERATE_OPS)
        if accelerate_cmds:
            self.last_accelerate_op = accelerate_cmds[0]

    def get_last_commands(self):
        return self.commads

    def stop(self):
        self.velocity = np.float32(0.0)
        self.acceleration = np.float32(0.0)

    def delta_time(self):
        return self.states[-1].t - self.states[0].t

    def to_reward_values(self):
        return {
            settings.REWARD_KEY_POS: self.pos,
            settings.REWARD_KEY_ANGLE: self.angle,
            settings.REWARD_KEY_VELOCITY: self.velocity,
            settings.REWARD_KEY_TIME: self.last_time
        }

    pass  # class VehicleState


class Vehicle:
    def __init__(self, name):
        self._name = name
        self._state = VehicleState()
        self._reward = Reward()
        self._vehicle_id = self._reward.get_next_id()
        self._drive_time = np.float32(0.0)

    @property
    def name(self):
        return self._name

    @property
    def vehicle_id(self):
        return self._vehicle_id

    @property
    def front(self):
        return self._state.front

    @property
    def acceleration(self):
        return self._state.acceleration

    @property
    def velocity(self):
        return self._state.velocity

    def _set_small_velocity(self):
        self._state.velocity = settings.VEHICLE.TRANSMISSION_SPEED[-1]*settings.VEHICLE.SMALL_SPEED_COEF

    @property
    def rotation(self):
        return self._state.angle

    @property
    def wheel_angle(self):
        return self._state.wheel_angle

    @property
    def wheel_delta_angle(self):
        return self._state.wheel_delta_angle

    @property
    def position(self):
        return copy.deepcopy(self._state.pos)

    @property
    def state(self):
        return copy.deepcopy(self._state)

    @state.setter
    def state(self, s):
        self._state = copy.deepcopy(s)

    @property
    def saved_interval(self):
        return self._state.delta_time()

    def reset(self):
        self._state.stop()

    def initialize(self, pos: Point, angle, velocity, lines: List[Line] = None):
        self.reset()
        self._state.clear()
        if (lines is not None) and self._check_collision(pos, np.float32(angle), lines):
            return False
        self._state.replace(pos, angle, np.float32(0.0))
        self._state.velocity = np.float32(velocity)
        self._drive_time = np.float32(0.0)
        return True

    def setup_reward(self):
        self._reward.set_values(self._vehicle_id, self._state.to_reward_values())

    def commands(self, cmds: List[VehicleOp]):
        if not cmds:
            return
        self.setup_reward()
        if settings.VERBOSE_VEHICLE_COMMANDS:
            print("Commands:")
        for cmd_key in cmds:
            cmd = VEHICLE_SINGLE_OPERATIONS[cmd_key]
            if settings.VERBOSE_VEHICLE_COMMANDS:
                print("\t{}".format(cmd.name))
            if cmd.forward is not None:
                self.command_transmission(cmd.forward)
                continue
            if cmd.break_value is not None:
                self.command_backward(cmd.break_value)
            elif cmd.accelaration_value is not None:
                self.command_accelerate(cmd.accelaration_value)
            if cmd.rotate is not None:
                self.command_turn(cmd.rotate)
        self._state.set_commands(cmds)

    # def _can_use_transmite_commands(self):
    #    return np.isclose(self._state.velocity, np.float32(0.0), atol=1e-1)

    def get_next_available_ops(self, filter_ops: bool = False):
        cmds = [(VehicleOp.NONE,)]
        # if self._can_use_transmite_commands():
        #    cmds.extend([(o,) for o in VEHICLE_TRANSMITE_OPS])
        for rot_op in VEHICLE_ROTATE_OPS:
            for angle_op in VEHICLE_ACCELERATE_OPS:
                # if filter_ops:
                #    if (self._state.last_rotate_op is not None
                #        and abs(self._state.last_rotate_op.value - rot_op.value) > 1
                #        ):
                #        continue
                #    if (self._state.last_accelerate_op is not None
                #        and abs(self._state.last_accelerate_op.value - angle_op.value) > 1
                #        ):
                #        continue
                if (
                    (rot_op.value == VehicleOp.ROTATE_NONE.value)
                    and (angle_op.value == VehicleOp.ACCELERATE_NONE.value)
                    ):
                    continue
                # if (np.isclose(self._state.velocity, np.float32(0.0))
                #    and (angle_op.value < VehicleOp.ACCELERATE_NONE.value)
                #    ):
                #    continue
                cmds.append((rot_op, angle_op))
        return cmds

    def command_transmission(self, front: bool):
        """
        set up transmission of vehicle
        :param front: True or False - move forward or backward
        :return: True if command have been done ok
        """
    #    if self._can_use_transmite_commands():
    #        self._state.front = front
    #        return True
    #    elif VEHICLE.VERBOSE:
    #        print("{} is not stoped to switch transmission to {}"
    #              .format(self._name, "forward" if front else "backward"))
    #        return False
        pass

    def get_backward_count(self):
        """
        :return: break commands count [0, get_break_count()]
        """
        return get_breaks_count()

    def get_acceleration_count(self):
        """
        :return: acceleration commands count [0, get_acceleration_count()]
        """
        return get_accelaration_count()

    def command_backward(self, value: int):
        """
        implement break command
        :param value: [0, get_break_count()] possible commands to lower speed of vehicle
        :return: True if command have been done ok
        """
        bc = self.get_backward_count()
        if (value < 0) or (value > bc):
            return False
        self._state.acceleration = settings.VEHICLE.TRANSMISSION_ACCELERATION[bc - value]
        return True

    def command_accelerate(self, value: int):
        """
        :param value: [0, get_acceleration_count()] possible commands to accelerate vihicle
        :return: True if command done have been ok
        """
        bc = self.get_backward_count()
        ac = self.get_acceleration_count()
        if (value < 0) or (value > ac):
            return False
        self._state.acceleration = settings.VEHICLE.TRANSMISSION_ACCELERATION[value + bc]
        return True

    def get_rotate_count(self):
        """
        :return: rotate commands count [-get_rotate_count(), get_rotate_count()]
        """
        return settings.VEHICLE.ROTATE_ACCELERATION.shape[0] // 2

    def command_turn(self, value: int):
        """
        :param value: [-get_rotate_count(), get_rotate_count()] command to turn
        :return: True if command have been done ok
        """
        rc = self.get_rotate_count()
        if (value < -rc) or (value > rc):
            return False
        self._state.wheel_delta_angle = settings.VEHICLE.ROTATE_ACCELERATION[rc + value]
        return True

    def _velocity_step(self, t: np.float32):
        self._state.velocity += self._state.acceleration * t  # * np.float32(1.0 if self.front else -1.0)
        assert(settings.VEHICLE.TRANSMISSION_SPEED[-1] > 0)
        if self._state.velocity > settings.VEHICLE.TRANSMISSION_SPEED[-1]:
            self._state.velocity = settings.VEHICLE.TRANSMISSION_SPEED[-1]
        assert(settings.VEHICLE.TRANSMISSION_SPEED[0] < 0)
        if self._state.velocity < settings.VEHICLE.TRANSMISSION_SPEED[0]:
            self._state.velocity = settings.VEHICLE.TRANSMISSION_SPEED[0]
        return True

    def _turn_step(self, t: np.float32):
        self._state.wheel_angle += self._state.wheel_delta_angle * t
        assert((settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0] < 0) and (settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1] > 0))
        if self._state.wheel_angle < settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0]:
            self._state.wheel_angle = settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0]
        if self._state.wheel_angle > settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1]:
            self._state.wheel_angle = settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1]
        return True

    def _get_bounds(self, pos: Point, angle: np.float32):
        return [
            pt.rotate(settings.VEHICLE.CENTER_PT, angle) + pos
            for pt in settings.VEHICLE.BOUNDS_RECT
        ]

    def get_bounds(self):
        return self._get_bounds(self._state.pos, self._state.angle)

    def get_axles(self):
        pts = [
            pt.rotate(settings.VEHICLE.CENTER_PT, self._state.angle) + self._state.pos
            for pt in settings.VEHICLE.AXLE_FRONT
        ]
        pts.extend([
            pt.rotate(settings.VEHICLE.CENTER_PT, self._state.angle) + self._state.pos
            for pt in settings.VEHICLE.AXLE_BACK
        ])
        return pts

    def get_front_wheels(self):
        pts = [
            pt.rotate(settings.VEHICLE.CENTER_PT, self._state.angle) + self._state.pos
            for pt in settings.VEHICLE.AXLE_FRONT
        ]
        res = list()
        for pt in pts:
            angle = self._state.angle - self._state.wheel_angle
            pt1 = Point.vector_from_angle(angle) * settings.VEHICLE.WHEEL_SIZE + pt
            pt2 = Point.vector_from_angle(angle + math.pi) * settings.VEHICLE.WHEEL_SIZE + pt
            res.append(pt1)
            res.append(pt2)
        return res

    def _check_collision(self, pos: Point, angle: np.float32, lines: List[Line]):
        bounds_lines = get_lines(self._get_bounds(pos, angle))
        for bl in bounds_lines:
            for l in lines:
                b, v = Line.distance_line_line(pos, np.float32(0), bl, l)
                if b and math.fabs(v) < 1e-1:
                    return True
        return False

    def _move_linear(self, t: np.float32, lines: List[Line]):
        run_len = self._state.velocity * t
        pos = self._state.pos + Point.vector_from_angle(self._state.angle)*run_len
        if self._check_collision(pos, self._state.angle, lines):
            return False
        self._state.pos = pos
        return True

    def _move_with_turn(self, t: np.float32, lines: List[Line]):
        run_len = self._state.velocity * t
        if np.allclose(run_len, np.float32(0.0), np.float32(1e-5)):
            return True
        abpt = (settings.VEHICLE.AXLE_BACK[0] + settings.VEHICLE.AXLE_BACK[1]) * 0.5 + self._state.pos
        afpt = (settings.VEHICLE.AXLE_FRONT[0] + settings.VEHICLE.AXLE_FRONT[1]) * 0.5 + self._state.pos
        al = (abpt - afpt).length()
        rb = np.float32(al / math.tan(math.fabs(self._state.wheel_angle)))
        beta = run_len/rb
        beta = np.float32(-beta if self._state.wheel_angle >= 0 else beta)
        move_angle = (self._state.angle + beta*0.5)
        delta_move = Point.vector_from_angle(move_angle, r=run_len)
        pos = self._state.pos + delta_move
        angle_new = self._state.angle + beta
        if self._check_collision(pos, angle_new, lines):
            return False
        self._state.pos = pos
        self._state.angle = angle_new
        return True

    def get_drive_time(self):
        return self._drive_time

    def step(self, t, lines: List[Line]):
        if t > 0:
            self._drive_time += t
        res1, res2 = self._step(t, lines)
        if res1:
            nv = self._state.to_reward_values()
            nv[settings.REWARD_KEY_COLLISION] = not res2
            nv[settings.REWARD_KEY_DRIVE_TIME] = self._drive_time
            reward = self._reward.get_reward(self._vehicle_id, nv)
        else:
            reward = None
        return res1 and res2, reward

    def model_ops(self, ops, t):
        """
        this functionality could be implemented with "dream" neural network.
        this implementation just for speed optimization.
        """
        s = self.state
        v = Vehicle(name=self._name + "_model_ops")
        dt = v.get_drive_time()
        res = list()
        for cmd in ops:
            v.state = s
            v._drive_time = dt
            v.commands(cmd)
            v.step(t, [])  # do not check for collision in dream functionality
            angle_norm, velocifty_norm = v.get_last_operation_info()
            if not np.allclose(velocifty_norm, np.float32(0.0),
                               settings.VEHICLE.TRANSMISSION_SPEED[-1]*settings.VEHICLE.SMALL_SPEED_COEF
                               ):
                res.append((angle_norm, velocifty_norm))
            else:
                v._drive_time = dt
                v._set_small_velocity()
                v.setup_reward()
                v.step(t, [])
                angle_norm, _ = v.get_last_operation_info()
                res.append((angle_norm, velocifty_norm))
        return res

    def _step(self, t, lines: List[Line]):
        """
        to process one time step -- one moment change for speed, rotation, position
        :param t: time interval
        :return: if step was ok
        """
        t = np.float32(t)
        if t <= 0:
            return False, False
        self._state.add(t)
        if not self._velocity_step(t):
            return False, False
        if not self._turn_step(t):
            return False, False
        if np.allclose(self._state.wheel_angle, 0.0, 1e-2):
            return True, self._move_linear(t, lines)
        else:
            return True, self._move_with_turn(t, lines)

    def get_input(self, env: StaticEnv):
        all_lines = env.get_lines()
        inputs = list()
        for index in settings.VEHICLE_STATE_INDEXES:
            inp = list()
            p = self._state.get_at(index)
            pt, angle0 = copy.deepcopy(p.pos), p.angle
            for ai in range(0, 360, settings.ANGLE_FOR_INPUT):
                angle = np.float32(math.pi*ai/180.0 + angle0)
                dist = [Line.distance_line_pt(pt, angle, l, pt) for l in all_lines]
                dist = [np.float32(math.fabs(d[1])) for d in dist if d[0]]
                dist.append(settings.MAX_INPUT_LINE_LENGTH)
                lmin = min(dist)
                pt2 = Point.vector_from_angle(angle, lmin) + pt
                inp.append(Line(pt, pt2))
            inputs.append(inp)
        return inputs

    def get_input_coefs_from_inputs(self, inputs):
        return [
                [   np.float32(
                        math.log(l.length - settings.MIN_INPUT_LINE_LENGTH + np.float32(1.0))
                        /math.log(settings.MAX_INPUT_LINE_LENGTH - settings.MIN_INPUT_LINE_LENGTH + np.float32(1.0))
                        - 0.5)
                    for l in inp
                ]
                for inp in inputs
            ]

    def get_input_coefs(self, env: StaticEnv):
        return self.get_input_coefs_from_inputs(self.get_input(env))

    def get_sensor_inputs(self):
        si = np.zeros((settings.SENSOR_INPUTS_SIZE,), dtype=np.float32)
        si[0] = np.float32(
            (self._state.velocity - settings.VEHICLE.TRANSMISSION_SPEED[0])
            / (settings.VEHICLE.TRANSMISSION_SPEED[-1] - settings.VEHICLE.TRANSMISSION_SPEED[0])
            - 0.5
        )
        si[1] = np.float32(
            (self._state.acceleration - settings.VEHICLE.TRANSMISSION_ACCELERATION[0])
            / (settings.VEHICLE.TRANSMISSION_ACCELERATION[-1] - settings.VEHICLE.TRANSMISSION_ACCELERATION[0])
            - 0.5
        )
        si[2] = np.float32(
            (self.wheel_angle - settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0])
            / (settings.VEHICLE.ROTATE_ANGLE_BOUNDS[-1] - settings.VEHICLE.ROTATE_ANGLE_BOUNDS[0])
            - 0.5
        )
        si[3] = np.float32(
            (self.wheel_delta_angle - settings.VEHICLE.ROTATE_ACCELERATION[0])
            / (settings.VEHICLE.ROTATE_ACCELERATION[-1] - settings.VEHICLE.ROTATE_ACCELERATION[0])
            - 0.5
        )
        for i, index in enumerate(settings.VEHICLE_STATE_INDEXES):
            p = self._state.get_at(index)
            pt, angle0 = copy.deepcopy(p.pos), p.angle
            angle0 = math.fmod(angle0, math.pi*2.0)
            angle0 = math.fmod(angle0 + math.pi*2.0, math.pi*2.0)
            si[4 + 3*i + 0] = np.float32(angle0/(math.pi*2.0) - 0.5)
            si[4 + 3*i + 1] = np.float32(-0.5 if pt.y < 0 else 0.5)
            pt *= np.float32(1.0/settings.TARGET.TARGET_POINT_MAX_DISTANCE)
            if pt.length() > 1:
                pt *= np.float32(1.0/pt.length())
            si[4 + 3*i + 2] = np.float32(pt.x * 0.5)

        return si

    def get_last_operation_info(self):
        p_current = self._state.get_at(0)
        p_prev = self._state.get_at(1)
        current_angle = math.fmod(p_current.angle, np.float32(math.pi*2.0))
        prev_angle = math.fmod(p_prev.angle, np.float32(math.pi*2.0))
        current_angle = math.fmod(current_angle + np.float32(math.pi*2.0), np.float32(math.pi*2.0))
        prev_angle = math.fmod(prev_angle + np.float32(math.pi*2.0), np.float32(math.pi*2.0))
        sign = np.float32(1.0 if current_angle - prev_angle >= 0 else -1.0)
        delta_angle = angle_diff(current_angle, prev_angle)
        delta_angle = scipy.special.erf(delta_angle/np.sqrt(settings.ANGLE_FOR_OPERATION_MAX/25))
        delta_angle = delta_angle * sign
        delta_angle *= np.float32(0.5)  # [-0.5..0.5]
        velocity_norm = np.float32(
                (self._state.velocity - settings.VEHICLE.TRANSMISSION_SPEED[0])
                / (settings.VEHICLE.TRANSMISSION_SPEED[-1] - settings.VEHICLE.TRANSMISSION_SPEED[0])
                - 0.5)
        assert(delta_angle >= np.float32(-0.5) and delta_angle <= np.float32(0.5))
        assert(velocity_norm >= np.float32(-0.5) and velocity_norm <= np.float32(0.5))
        return delta_angle, velocity_norm

    pass  # class Vehicle


# RLParking/db.py

class StartCoord:
    def __init__(self, pos: Point, angle: np.float32, weight: int):
        self.pos = pos
        self.angle = angle
        self.weight = weight
        pass

    pass  # class StartCoord


class StartCoordinates:
    def __init__(self):
        self._start_positions = settings.START_POSITIONS

    def get_rand(self, count: int = 1):
        r = np.random.rand(2)
        p1 = Point.between(self._start_positions[0].pt0, self._start_positions[0].pt1, r[0])
        p2 = Point.between(self._start_positions[1].pt0, self._start_positions[1].pt1, r[0])
        p = Point.between(p1, p2, r[1])
        return [StartCoord(
            pos=p,
            angle=np.random.randint(0, 2)*np.float32(math.pi) + np.float32(math.pi*0.5),
            weight=1
            )
            for _ in range(count)]

    def get_pos_with_weights(self):
        return

    pass  # class StartCoordinates


@froze_class
class TrainState:
    def __init__(self,
            state: VehicleState, next_state: VehicleState,
            reward: np.float32, ops, values,
            final_state: bool
        ):
        self.state = state
        self.next_state = next_state
        self.reward = reward
        self.final_state = final_state
        self.ops = ops
        self.values = values
        self.getted = 0

    def has_reward(self):
        if settings.USE_LEARNING_BOOST:
            return self.reward >= settings.REINFORCE_DONE * np.float32(0.8)
        else:
            return ((self.reward >= settings.REINFORCE_DONE * np.float32(0.8))
                or (self.reward <= settings.REINFORCE_FAIL * np.float32(0.8)))

    def get_reward_index(self, max_reward):
        if self.has_values():
            return 3
        elif self.reward >= max_reward * np.float32(0.8):
            return 2
        elif self.reward <= settings.REINFORCE_FAIL * np.float32(0.8):
            return 1
        else:
            return 0

    def has_values(self):
        return self.ops is not None if self.getted < settings.GAME_STATE_GETTED_MAX_COUNT else False

    pass  # class TrainState


class TrainStates:
    def __init__(self, max_length: int, reward_ratio: np.float32):
        self._states = list()
        self._max_length = max_length
        self._reward_ratio = reward_ratio
        if settings.PATHES.has_train_state():
            self._load(settings.PATHES.get_train_state())
        self._init_indexes()
        self._calc_avg_values()

    def __len__(self):
        return len(self._states)

    def can_train(self):
        return len(self._states) >= settings.GAME_STATES_IN_DB_TO_START_TRAIN

    def add_state(self,
            state: VehicleState, next_state: VehicleState,
            reward: np.float32, ops, values, final_state: bool
        ):
        self._states.append(TrainState(state, next_state, reward, ops, values, final_state))

    def set_reward(self, reward: np.float32, t: np.float32, vehicle_id: int):
        for s in self._states:
            if (s.t == np.float32(0.0)) or (s.vehicle_id != vehicle_id):
                continue
            if t - s.t >= np.float32(0.0):
                s.reward += np.float32(math.pow(settings.RL_REWARD_COEF,
                                                (t - s.t)*settings.VEHICLE_UI_BOT_STEPS_PER_SECOND)
                                       * reward)
            s.t = np.float32(0.0)

    def _calc_avg_values(self):
        if self._states:
            self._avg_got = sum([s.getted for s in self._states])/len(self._states)
            self._avg_reward = sum([s.reward for s in self._states])/len(self._states)
            self._avg_values = sum([1 if s.has_values() else 0 for s in self._states])/len(self._states)
        else:
            self._avg_got = 0.0
            self._avg_reward = 0.0
            self._avg_values = 0.0

    @staticmethod
    def has_reward(s: TrainState, max_reward, inverse=False):
        v = s.get_reward_index(max_reward)
        if inverse:
            v = 9 - v
        return v

    def reduce_size(self):
        if len(self._states) <= self._max_length:
            return
        self._calc_avg_values()
        random.shuffle(self._states)
        self._init_max_reward()
        self._states = radix_sorted(
            self._states, 1,
            key=partial(TrainStates.has_reward, max_reward=self._max_reward, inverse=True)
        )
        length = int(self._reward_ratio*len(self._states))
        more = self._states[length:]
        self._states = self._states[:length]
        random.shuffle(more)
        self._states = self._states + more
        self._states = self._states[:self._max_length]
        self._init_indexes()
        if settings.VERBOSE_TRAIN_DB:
            print("len(Train_DB_states)={} avg_getted={:.6f} avg_reward = {:.6f} avg_values = {:0.6f}"
                .format(len(self._states), self._avg_got, self._avg_reward, self._avg_values)
                )

    def print_size(self):
        self._calc_avg_values()
        if settings.VERBOSE_TRAIN_DB:
            print("len(Train_DB_states)={} avg_getted={:.6f} avg_reward = {:.6f} avg_values = {:0.6f}"
                .format(len(self._states), self._avg_got, self._avg_reward, self._avg_values)
                )

    def _init_max_reward(self):
        rewards = [s.reward for s in self._states]
        if rewards:
            self._max_reward = max(rewards)
            self._max_reward = max([self._max_reward, settings.REINFORCE_DONE*0.2])
        else:
            self._max_reward = settings.REINFORCE_DONE * 0.2

    def _init_indexes(self):
        self._init_max_reward()
        self._indexes = np.array(
            radix_sorted_indexes(
                self._states, 1,
                key=partial(TrainStates.has_reward, max_reward=self._max_reward, inverse=True)
            ),
            dtype=np.int32)
        self._reward_count = sum(
            [1 if TrainStates.has_reward(s, max_reward = self._max_reward)>0 else 0
            for s in self._states]
        )

    def get_items(self, count: int = 1):
        if self._indexes.shape[0] != len(self._states):
            self._init_indexes()
        items_count = len(self._states)
        if self._reward_count > 0:
            p1 = max((self._reward_count/items_count, self._reward_ratio))
        else:
            p1 = self._reward_count/items_count
        p2 = (1.0 - p1)
        if items_count - self._reward_count > 0:
            p2 /= (items_count - self._reward_count)
        if self._reward_count > 0:
            p1 /= self._reward_count
        p = np.zeros((items_count,), dtype=np.float32)
        p[:self._reward_count] = np.float32(p1)
        p[self._reward_count:] = np.float32(p2)
        indexes = np.random.choice(self._indexes, count, replace=False, p=p)
        for i in range(count):
            self._states[indexes[i]].getted += 1
        return [self._states[indexes[i]] for i in range(count)]

    def get_all_items(self):
        return self._states

    def save(self, fn: AnyStr):
        with open(fn, "wb") as f:
            rp = settings.TARGET.get_reinforce_params()
            print("db save reinforce params = {}".format(rp))
            data = rp, self._states
            pickle.dump(data, f)

    def _load(self, fn: AnyStr):
        self._states = list()
        with open(fn, "rb") as f:
            reinforce_params, self._states = pickle.load(f)
            settings.TARGET.set_reinforce_params(reinforce_params)

    pass  # class TrainStates


class AsyncTrainDBProcessor:
    def __init__(self, db: TrainStates):
        self._can_process_requests = [RequestType.SAVE_TRAIN_STATE]
        self._db = db

    def process(self, request: Request):
        if not request.in_types(self._can_process_requests):
            return False
        if request.is_type(RequestType.SAVE_TRAIN_STATE):
            self._db.add_state(
                request.state, request.next_state,
                np.float32(request.reward), request.ops, request.values,
                request.final_state
            )
        return True

    pass  # class AsyncTrainDBProcessor



# RLParking/bot.py

class AsyncVehicleBot:
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 600
    SCREEN_CENTER = Point(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)

    def __init__(self, ui_bot: bool, start_coordinates: StartCoordinates, env: StaticEnv):
        self._ui_bot = ui_bot
        if self._ui_bot:
            pass

        self._vehicle = Vehicle("UIBotVehicle" if self._ui_bot else "BotVehicle")
        self._done = False
        self._env = env
        self._start_coordinates = start_coordinates
        self._ops = list()
        self._ops_value = np.zeros((1,), dtype=np.float32)
        self._initialize()

    def _initialize(self):
        # print("start vehicle {} -- {}".format(self._vehicle.vehicle_id, self._vehicle.name))
        sc = self._start_coordinates.get_rand()[0]
        self._vehicle.initialize(
            pos=sc.pos,
            angle=sc.angle,
            velocity=0.0
        )
        self._inputs = self._vehicle.get_input(self._env)

    def _get_lines(self):
        return self._env.get_lines()

    def _c(self, pt, c):
        return Point(pt.x*c.x, -pt.y*c.y) + self.SCREEN_CENTER

    def _draw_line(self, line: Line, coefs, color=(128, 128, 128)):
        pass

    def _draw_rect(self, rc: Rect, coefs, color=(128, 128, 128)):
        pass

    def _draw_input(self, coefs):
        pass

    @staticmethod
    def _has(all_op_lists, op_list):
        for i, op_l in enumerate(all_op_lists):
            if len(op_l) != len(op_list):
                continue
            if all([op_1.value == op_2.value for op_1, op_2 in zip(op_l, op_list)]):
                return i
        return -1

    def _draw_net_results(self, ops, ops_values, coefs):
        pass

    def _draw_dist_ltl(self, coefs):
        pass

    def _draw_target(self, coefs):
        pass

    def _draw_weights(self, coefs):
        pass

    def _draw_vehicle(self, coefs):
        pass

    def _draw(self, ops, ops_values):
        pass

    def process(self):
        try:
            self._ops = list()
            self._ops_value = np.zeros((1,), dtype=np.float32)
            self._initialize()
            t0 = time.time()
            while True:
                yield from self._process_main()
                if self._ui_bot:
                    t1 = time.time()
                    while (
                        int(t1/1.5)  # *VEHICLE_UI_BOT_STEPS_PER_SECOND
                        == int(t0/1.5)  # *VEHICLE_UI_BOT_STEPS_PER_SECOND
                        ):
                        _ = yield Request(RequestType.NOP)
                        t1 = time.time()
                    t0 = t1
                    self._draw(self._ops, self._ops_values)
                pass
        except StopIteration as e:
            return

    def _set_operations_values(self, ops, ops_values):
        p = np.copy(ops_values.flatten())
        p -= np.min(p) - np.float32(0.01)
        s = np.sum(p)
        if s > 0:
            p /= s
        else:
            p[:] = np.float32(1.0/p.shape[0])
        indexes = np.random.choice(p.shape[0], min((settings.STATE_LEARN_VALUES_COUNT, p.shape[0])), replace=False, p=p)
        values = np.zeros(indexes.shape, dtype=np.float32)
        selected_ops = list()
        selected_commands = list()
        for i in range(indexes.shape[0]):
            index = indexes[i]
            v = Vehicle("test_values")
            v.state = self._vehicle.state
            pos_end = pos_start = v.state.get_at(0).pos
            steps = 0
            op = ops[index]
            selected_commands.append(op)
            v.commands(op)
            last_op_ok, reward = v.step(t=settings.VEHICLE_STEP_DURATION, lines=self._env.get_lines())
            final_state = not ((not reward[1]) and last_op_ok)
            reward_value = reward[0]
            steps += 1
            selected_ops.append(v.get_last_operation_info())
            while not final_state:  # and (reward_value == np.float32(0.0))
                ops1 = v.get_next_available_ops()
                ops1_inputs = v.model_ops(ops1, t=settings.VEHICLE_STEP_DURATION)
                inputs = v.get_input_coefs(self._env)
                inputs = np.array(inputs, dtype=np.float32)
                sensor_inputs = v.get_sensor_inputs()
                inputs = np.concatenate((inputs.flatten(), sensor_inputs))
                request = yield Request(
                    type=RequestType.GET_BEST_OPERATION,
                    inputs=inputs,
                    ops=ops1_inputs
                )
                op = ops1[request.get_best_operation()]
                v.commands(op)
                last_op_ok, reward = v.step(t=settings.VEHICLE_STEP_DURATION, lines=self._env.get_lines())
                final_state = not ((not reward[1]) and last_op_ok)
                reward_value = reward[0]
                pos_end = v.state.get_at(0).pos
                steps += 1
            drive_length = (pos_end - pos_start).length()
            if reward_value < 0:
                c1 = math.pow(settings.RL_RO_COEF, steps)
                c2 = math.pow(settings.RL_RO_COEF,
                              drive_length/(settings.VEHICLE.TRANSMISSION_SPEED[-1]*np.float32(0.5)))
                c_reward = max((c1,c2))
            else:
                c_reward = math.pow(settings.RL_RO_COEF, steps)
            values[i] = np.float32(c_reward*reward_value)
        if not selected_ops:
            return None, None, None
        else:
            return selected_commands[np.argmax(values)], selected_ops, values

    def _process_main(self):
        ops = self._vehicle.get_next_available_ops()
        ops_inputs = self._vehicle.model_ops(ops, t=settings.VEHICLE_STEP_DURATION)
        self._inputs = self._vehicle.get_input(self._env)
        inputs = self._vehicle.get_input_coefs_from_inputs(self._inputs)
        inputs = np.array(inputs, dtype=np.float32)
        sensor_inputs = self._vehicle.get_sensor_inputs()
        inputs = np.concatenate((inputs.flatten(), sensor_inputs))
        request = yield Request(
            type=RequestType.GET_BEST_OPERATION,
            inputs=inputs,
            ops=ops_inputs
        )

        op = ops[request.get_best_operation()]
        if not self._ui_bot and (np.random.rand(1)[0] <= settings.SET_OPERATION_VALUES_PROB):
            op, selected_ops, selected_ops_values = yield from self._set_operations_values(ops, request.results)
        else:
            selected_ops, selected_ops_values = None, None
        if not selected_ops:
            op = ops[request.get_best_operation()]

        self._ops = ops
        self._ops_values = request.results
        assert(len(self._ops) == self._ops_values.shape[0])
        # setup commands
        self._vehicle.commands(op)
        state = self._vehicle.state
        # run commands
        last_op_ok, reward = self._vehicle.step(t=settings.VEHICLE_STEP_DURATION, lines=self._env.get_lines())
        final_state = not ((not reward[1]) and last_op_ok)
        _ = yield Request(
            type=RequestType.SAVE_TRAIN_STATE,
            state=state,
            next_state=self._vehicle.state,
            reward=reward[0],
            ops=selected_ops,
            values=selected_ops_values,
            final_state=final_state
        )
        if self._ui_bot:
            self._draw(self._ops, self._ops_values)
        if not ((not reward[1]) and last_op_ok):
            self._initialize()

    pass  # class AsyncVehicleBot



# RLParking/train.py


class AsyncTrain:
    def __init__(self, db: TrainStates, env: StaticEnv):
        self._db = db
        self._env = env

    def process(self):
        try:
            db_size = 0
            while len(self._db) < settings.GAME_STATES_IN_DB_TO_START_TRAIN:
                _ = yield Request(RequestType.NOP)
                if len(self._db) >= db_size + settings.GAME_STATES_IN_DB_STEP_SIZE:
                    self._db.print_size()
                    db_size = ((len(self._db) // settings.GAME_STATES_IN_DB_STEP_SIZE)
                              * settings.GAME_STATES_IN_DB_STEP_SIZE)

            vehicle = Vehicle("TrainVehicle")
            while True:
                yield from self._process_main(vehicle)

                # output db size
                if len(self._db) >= db_size + settings.GAME_STATES_IN_DB_STEP_SIZE:
                    self._db.print_size()
                    db_size = ((len(self._db) // settings.GAME_STATES_IN_DB_STEP_SIZE)
                               * settings.GAME_STATES_IN_DB_STEP_SIZE)
                # check states db size
                if len(self._db) > settings.GAME_STATES_IN_DB_SIZE_MAX2:
                    self._db.reduce_size()
                    self._db.save(settings.PATHES.get_train_state())
                    db_size = len(self._db)
        except StopIteration as e:
            return

    def _prepare_inputs(self, states: List[TrainState], vehicle: Vehicle):
        inputs = list()
        ops = list()
        for train_state in states:
            vehicle.state = train_state.state
            state_inputs = vehicle.get_input_coefs(self._env)
            state_inputs = np.array(state_inputs, dtype=np.float32)
            sensor_inputs = vehicle.get_sensor_inputs()
            state_inputs = np.concatenate((state_inputs.flatten(), sensor_inputs))
            inputs.append(state_inputs)
            # ops.append(train_state.state.get_last_commands())
            ops.append(vehicle.get_last_operation_info())
        return inputs, ops

    def _prepare_values(self, states: List[TrainState], vehicle: Vehicle):
        values = np.zeros((len(states),), dtype=np.float32)
        for i,train_state in enumerate(states):
            if train_state.final_state:
                values[i] = train_state.reward
            else:
                vehicle.state = train_state.next_state
                ops = vehicle.get_next_available_ops()
                ops_inputs = vehicle.model_ops(ops, settings.VEHICLE_STEP_DURATION);
                inputs = vehicle.get_input_coefs(self._env)
                inputs = np.array(inputs, dtype=np.float32)
                sensor_inputs = vehicle.get_sensor_inputs()
                inputs = np.concatenate((inputs.flatten(), sensor_inputs))
                request = yield Request(
                    type=RequestType.GET_BEST_OPERATION,
                    inputs=inputs,
                    ops=ops_inputs
                )
                values[i] = (
                    request.get_best_operation_value()*settings.VALUES_TO_REWARD_COEF*settings.RL_RO_COEF
                    + train_state.reward
                    )
        return np.clip(values * settings.REWARD_TO_VALUES_COEF, np.float32(-0.5), np.float32(0.5))

    def _prepare_inputs_values2(self, states: List[TrainState], vehicle: Vehicle):
        inputs = list()
        values = list()
        ops = list()
        for i, train_state in enumerate(states):
            if train_state.has_values():
                vehicle.state = train_state.state
                state_inputs = vehicle.get_input_coefs(self._env)
                state_inputs = np.array(state_inputs, dtype=np.float32)
                sensor_inputs = vehicle.get_sensor_inputs()
                state_inputs = np.concatenate((state_inputs.flatten(), sensor_inputs))
                for op in train_state.ops:
                    inputs.append(state_inputs)
                    ops.append(op)
                values.append(train_state.values)
        if values:
            values = np.stack(values).flatten()
            values = np.clip(values * settings.REWARD_TO_VALUES_COEF, np.float32(-0.5), np.float32(0.5))
        return inputs, ops, values

    def _process_main(self, vehicle: Vehicle):
        states = self._db.get_items(count=settings.STATES_TO_TRAIN)
        inputs, ops = self._prepare_inputs(states, vehicle)
        values = yield from self._prepare_values(states, vehicle)
        _ = yield Request(
            type=RequestType.RL_TRAIN,
            inputs=inputs,
            ops=ops,
            values=values
        )
        inputs, ops, values = self._prepare_inputs_values2(states, vehicle)
        if inputs:
            _ = yield Request(
                type=RequestType.RL_TRAIN,
                inputs=inputs,
                ops=ops,
                values=values
            )

    pass  # class AsyncTrain


# DNN/net.py


class NetData:
    def __init__(self, batch):
        self.x = batch
        self.y = None

    pass  # class NetData


class NetTrainData(NetData):
    def __init__(self, batch_x, batch_y):
        super(NetTrainData, self).__init__(batch_x)
        self.y = batch_y

    pass  # class NetTrainData


class OperationLayer(tf.layers.Layer):
    """
    layer to present operation in the net data
    """

    def __init__(self, items_count, item_sz, extra_sz, **kwargs):
        self._items_count = items_count
        self._item_sz = item_sz
        self._extra_sz = extra_sz
        self._expand_op = None
        self._op_base = None
        super(OperationLayer, self).__init__(**kwargs)
        return

    def build(self, input_shape):
        a = np.zeros((self._items_count,), dtype=np.float32)
        a[:] = np.float32(0.5)
        e = np.zeros((self._items_count, self._items_count*self._item_sz), dtype=np.float32)
        for i in range(self._items_count):
            e[i, i*self._item_sz:(i+1)*self._item_sz] = np.float32(1.0)
        self._expand_op = tf.constant(e)
        self._op_base = tf.constant(a)
        super(OperationLayer, self).build(input_shape)
        return

    def call(self, inputs, op, **kwargs):
        """
        to implement layer operation
        :param inputs: input tensor
        :param op: operation numpy array
        :param kwargs: another arguments
        :return: result of operation
        """
        if self._extra_sz > 0:
            i1 = tf.slice(inputs, [0, 0], [-1, self._items_count * self._item_sz])
            i2 = tf.slice(inputs,
                          [0, self._items_count * self._item_sz],
                          [-1, self._extra_sz]
                          )
            op = tf.add(op, self._op_base)
            op1 = tf.matmul(op, self._expand_op)
            return tf.concat([tf.multiply(i1, op1), i2], 1)
        else:
            op = tf.add(op, self._op_base)
            op1 = tf.matmul(op, self._expand_op)
            return tf.multiply(inputs, op1)

    def compute_output_shape(self, input_shape):
        """
        to computer output shape from input shape
        :param input_shape:
        :return:
        """
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self._items_count * self._item_sz + self._extra_sz
        return tf.TensorShape(shape)

    def get_config(self):
        """
        to serialize layer data
        :return: layer config
        """
        base_config = super(OperationLayer, self).get_config()
        base_config['items_count'] = self._items_count
        base_config['item_sz'] = self._item_sz
        base_config['extra_sz'] = self._extra_sz
        return base_config

    @classmethod
    def from_config(cls, config):
        """
        create layer from the serialized data
        :param config: data to unserialize from
        :return:
        """
        return cls(**config)

    pass  # class OperationLayer


class RLModel(tf.keras.Model):
    """
    RL model class
    """
    def __init__(self):
        super(RLModel, self).__init__(name="parking_model")
        self._input_size = settings.NET_INPUT_SIZE
        self._op_count = settings.OPERATIONS_COUNT
        assert(settings.NET_OPERATION_EXTRA_SIZE >= 0)
        self._layers1 = list()
        self._layers1.append(tf.layers.Dense(settings.NET1_FC_SIZE1, activation=settings.NET_LAYER1_ACTIVATION))
        self._layers1.append(tf.layers.Dense(settings.NET1_FC_SIZE1, activation=settings.NET_LAYER1_ACTIVATION))
        self._layers1.append(tf.layers.Dropout(settings.NET1_FC_DROPOUT_VALUE1))
        self._layers1.append(tf.layers.Dense(settings.NET1_FC_SIZE1, activation=settings.NET_LAYER1_ACTIVATION))
        self._fc_size = self._op_count * settings.NET_OPERATION_ITEM_SIZE + settings.NET_OPERATION_EXTRA_SIZE
        self._layers1.append(tf.layers.Dense(self._fc_size, activation=settings.NET_LAYER1_ACTIVATION))
        self._op_layer = OperationLayer(
            items_count=self._op_count,
            item_sz=settings.NET_OPERATION_ITEM_SIZE,
            extra_sz=settings.NET_OPERATION_EXTRA_SIZE
        )
        self._layers2 = list()
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE2, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE2, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dropout(settings.NET1_FC_DROPOUT_VALUE2))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE3, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE3, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dropout(settings.NET1_FC_DROPOUT_VALUE2))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE3, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(settings.NET1_FC_SIZE3, activation=settings.NET_LAYER2_ACTIVATION))
        self._layers2.append(tf.layers.Dense(1))
        pass

    def call(self, inputs):
        x = tf.slice(inputs, [0, 0], [-1, self._input_size - self._op_count])
        op = tf.slice(inputs, [0, self._input_size - self._op_count], [-1, self._op_count])
        x = tf.reshape(x, [-1, self._input_size - self._op_count])
        for l in self._layers1:
            x = l(x)
        x = self._op_layer(x, op=op)
        for l in self._layers2:
            x = l(x)
        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = 1
        return tf.TensorShape(shape)

    @property
    def layers_p1(self):
        return self._layers1

    @property
    def layers_p2(self):
        return self._layers2

    pass  # RLModel


# RLParking/asyncnet.py


class AsyncNet:
    def __init__(self, db: TrainStates):
        self._can_process_requests = [
            RequestType.GET_BEST_OPERATION,
            RequestType.SEARCH_OPERATION,
            RequestType.RL_TRAIN
        ]
        self._init_model()
        self._last_save_t = time.time()
        self._db = db

    def _init_model(self):
        self._model = RLModel()
        self._model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=1e-2, momentum=0.9),
                            loss='mse', metrics=['mse'])

        if settings.PATHES.has_main_model():
            # https://github.com/tensorflow/tensorflow/issues/24623
            input_size = settings.NET_INPUT_SIZE + settings.OPERATIONS_COUNT*settings.NET_OPERATION_ITEM_SIZE
            x = np.zeros((settings.STATES_TO_TRAIN_BATCH_SIZE, input_size), dtype=np.float32)
            y = np.zeros((settings.STATES_TO_TRAIN_BATCH_SIZE, 1), dtype=np.float32)
            self._model.fit(
                x, y,
                epochs=1, batch_size=settings.STATES_TO_TRAIN_BATCH_SIZE, verbose=0,
                callbacks=None
            )

        if settings.PATHES.has_main_model():
            self._model.load_weights(settings.PATHES.get_main_model())
            print("model loaded {0}".format(settings.PATHES.get_main_model()))
        #    coef1 = tf.Variable(settings.RESET_WEIGHTS_COEF.RESET_WEIGHTS_COEF)
        #    tf.keras.backend.get_session().run(coef1.initializer)
        #    for l in self._model.layers_p1:
        #        for w in l.trainable_weights:
        #            assign1 = w.assign(w*coef1)
        #            tf.keras.backend.get_session().run(assign1)
        #    for l in self._model.layers_p2:
        #        for w in l.trainable_weights:
        #            assign1 = w.assign(w*coef1)
        #            tf.keras.backend.get_session().run(assign1)

    def _get_values(self, nd: NetData):
        return self._model.predict(nd.x, nd.x.shape[0])  # - np.float32(0.5)

    def _train(self, ntd: NetTrainData, batch_size):
        t = time.time()
        callbacks = list()
        if t > self._last_save_t + settings.MAIN_MODEL_SAVE_INTERVAL:
            self._model.save_weights(filepath=settings.PATHES.get_main_model(), overwrite=True, save_format="h5")
            self._last_save_t = t
            if settings.VERBOSE_MAIN_MODEL_SAVE:
                print("saved model {0}".format(settings.PATHES.get_main_model()))
        return self._model.fit(
            ntd.x, ntd.y,
            epochs=1, batch_size=batch_size, verbose=0,
            callbacks=callbacks if callbacks else None,
            use_multiprocessing=True,
            workers=2,
            )

    @staticmethod
    def _get_net_data_for_run(inputs, ops):
        inputs_np = np.array(inputs, dtype=np.float32).flatten()
        x = list()
        for op in ops:
            op_arr = np.full((settings.OPERATIONS_COUNT,), np.float32(-0.5), dtype=np.float32)
            op_arr[int((op[0] + 0.5)*settings.ANGLES_FOR_OPERATIONS)] = np.float32(0.5)
            op_arr[settings.ANGLES_FOR_OPERATIONS+0] = np.fabs(op[1])
            op_arr[settings.ANGLES_FOR_OPERATIONS+1] = 0.5 if op[1] >= np.float32(0.0) else -0.5
            # print("input: {}  {}".format(int((op[0] + 0.5)*settings.ANGLES_FOR_OPERATIONS), op[1]))
            x.append(np.concatenate((inputs_np, op_arr)))
        return NetData(np.stack(x))

    @staticmethod
    def _fill_result(request: Request, nd: NetData):
        request.results = nd.y.flatten()

    def _set_search_operation(self, nd: NetData):
        can_train = self._db.can_train()
        if ((not can_train and (np.random.rand() > settings.RL_PRETRAIN_SEARCH_COEF))
               or (can_train and (np.random.rand() > settings.RL_SEARCH_COEF))
            ):
            return
        if settings.RL_SEARCH_USE_ALPHA_ALGORITHM or (not can_train):
            v = np.random.randint(0, nd.y.shape[0], size=1, dtype=np.int32)
        else:
            p = nd.y.flatten()
            p -= np.min(p)
            s = np.sum(p)
            if s > 0:
                p /= s
            else:
                p[:] = np.float32(1.0)/p.shape[0]
            v = np.random.choice(nd.y.shape[0], 1, p=p)
        nd.y[v[0], :] = np.float32(settings.VALUE_MORE_THAN_MAX_NET_RESULT)

    @staticmethod
    def _get_train_data(inputs, ops, values):
        x = list()
        for input1, op in zip(inputs, ops):
            op_arr = np.full((settings.OPERATIONS_COUNT,), np.float32(-0.5), dtype=np.float32)
            op_arr[int((op[0] + 0.5) * settings.ANGLES_FOR_OPERATIONS)] = np.float32(0.5)
            op_arr[settings.ANGLES_FOR_OPERATIONS+0] = np.fabs(op[1])
            op_arr[settings.ANGLES_FOR_OPERATIONS+1] = 0.5 if op[1] >= np.float32(0.0) else -0.5
            input1_np = np.array(input1, dtype=np.float32).flatten()
            x.append(np.concatenate((input1_np, op_arr)))
        assert(len(x) == values.shape[0])
        return NetTrainData(np.stack(x), values)

    def process(self, request: Request):
        if not request.in_types(self._can_process_requests):
            return False
        if request.is_type(RequestType.GET_BEST_OPERATION):
            net_data = AsyncNet._get_net_data_for_run(request.inputs, request.ops)
            net_data.y = self._get_values(net_data)
            self._fill_result(request, net_data)
        elif request.is_type(RequestType.SEARCH_OPERATION):
            net_data = AsyncNet._get_net_data_for_run(request.inputs, request.ops)
            net_data.y = self._get_values(net_data)
            self._set_search_operation(net_data)
            AsyncNet._fill_result(request, net_data)
        elif request.is_type(RequestType.RL_TRAIN):
            net_train_data = AsyncNet._get_train_data(request.inputs, request.ops, request.values)
            self._train(net_train_data, settings.STATES_TO_TRAIN_BATCH_SIZE)
            pass
        return True

    pass  # class AsyncNet


# RLParking/main.py


class AsyncFunc:
    def __init__(self, agent_cls=None, agent_func=None):
        self._obj = agent_cls
        self._func = agent_func
        self._it = None

    def init(self):
        if self._obj is not None:
            self._it = self._obj.process()
        if self._func is not None:
            self._it = self._func()
        assert(self._it is not None)
        return next(self._it)

    def process(self, request_result: Request):
        return self._it.send(request_result)

    @classmethod
    def create_from_class(cls, agent_cls):
        return cls(agent_cls=agent_cls)

    @classmethod
    def create_from_func(cls, agent_func):
        return cls(agent_func=agent_func)

    pass  # class AsyncFunc


class Processor:
    def __init__(self, agent_cls=None, agent_func=None):
        self._obj = agent_cls
        self._func = agent_func

    def process(self, request: Request):
        if request.processed:
            return
        if self._obj is not None:
            processed = self._obj.process(request)
            request.processed = processed
        if self._func is not None:
            processed = self._func(request)
            request.processed = processed

    @classmethod
    def create_from_class(cls, agent_cls):
        return cls(agent_cls=agent_cls)

    @classmethod
    def create_from_func(cls, agent_func):
        return cls(agent_func=agent_func)

    pass  # class Processor


def process_command_line(argv):
    settings.RESET_WEIGHTS_COEF.clear()
    try:
        opts, args = getopt.getopt(argv, "r")
        for opt, arg in opts:
            if opt == '-r':
                settings.RESET_WEIGHTS_COEF.set()
    except getopt.GetoptError:
        pass


def main(argv):
    process_command_line(argv)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(sess)

    train_db = TrainStates(max_length=settings.GAME_STATES_IN_DB_SIZE_MAX,
                           reward_ratio=settings.GAME_STATES_IN_DB_REWARD_RATIO)

    if settings.PATHES.has_train_state():
        train_db.reduce_size()
        train_db.save(settings.PATHES.get_train_state())

    start_coordinates = StartCoordinates()
    env = StaticEnv()

    async_funcs = {
        AsyncFunc.create_from_class(
            agent_cls=AsyncVehicleBot(ui_bot=False, start_coordinates=start_coordinates, env=env)
        ): Request(RequestType.NOP),
        AsyncFunc.create_from_class(
            agent_cls=AsyncVehicleBot(ui_bot=True, start_coordinates=start_coordinates, env=env)
        ): Request(RequestType.NOP),
        AsyncFunc.create_from_class(
            agent_cls=AsyncTrain(db=train_db, env=env)
        ): Request(RequestType.NOP),
    }
    processors = [
        Processor.create_from_class(agent_cls=AsyncNet(db=train_db)),
        Processor.create_from_class(agent_cls=AsyncTrainDBProcessor(db=train_db))
    ]

    for f in async_funcs.keys():
        async_funcs[f] = f.init()

    while True:
        main_cycle(async_funcs, processors)
    pass


def main_cycle(async_funcs, processors):
    for f, r in async_funcs.items():
        if async_funcs[f].processed:
            async_funcs[f] = f.process(r)
        pass
    for p in processors:
        for r in async_funcs.values():
            if not r.processed:
                p.process(r)
    for r in async_funcs.values():
        if not r.processed:
            assert (r.is_type(RequestType.NOP))
            r.processed = True
    pass


if __name__ == "__main__":
    main(sys.argv[1:])

