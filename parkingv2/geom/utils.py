from functools import wraps, partial


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

    def __call__(self,*args,**kwargs):
        if self.instance == None:
            self.instance = self.klass(*args,**kwargs)
        return self.instance


def counting_sort_indexes(tlist_len, indexes_src, k, get_sortkey):
    """ Counting sort algo.
        Args:
            tlist: target list to sort
            k: max value assume known before hand
            get_sortkey: function to retrieve the key that is apply to elements of tlist to be used in the count list index.
            map info to index of the count list.
        Adv:
            The count (after cum sum) will hold the actual position of the element in sorted order
    """

    # Create a count list and using the index to map to the integer in tlist.
    if tlist_len==0:
        return list()

    count_list = [0] * (k)

    # iterate the tgt_list to put into count list
    for i in range(tlist_len):
        sortkey = get_sortkey(i)
        assert(sortkey < len(count_list))
        count_list[sortkey] += 1

    # Modify count list such that each index of count list is the combined sum of the previous counts
    # each index indicate the actual position (or sequence) in the output sequence.
    for i in range(1,k):
        count_list[i] += count_list[i-1]

    indexes = [0] * (tlist_len)
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
            get_sortkey: function to retrieve the key that is apply to elements of tlist to be used in the count list index.
            map info to index of the count list.
        Adv:
            The count (after cum sum) will hold the actual position of the element in sorted order
    """

    # Create a count list and using the index to map to the integer in tlist.
    count_list = [0] * (k)

    # iterate the tgt_list to put into count list
    for item in tlist:
        sortkey = get_sortkey(item)
        assert(sortkey < len(count_list))
        count_list[sortkey] += 1

    # Modify count list such that each index of count list is the combined sum of the previous counts
    # each index indicate the actual position (or sequence) in the output sequence.
    for i in range(1,k):
        count_list[i] += count_list[i-1]

    output = [None] * len(tlist)
    for i in range(len(tlist)-1, -1, -1):
        sortkey = get_sortkey(tlist[i])
        output[count_list[sortkey]-1] = tlist[i]
        count_list[sortkey] -= 1

    return output


def radix_sorted(tlist, k, key):
    def get_sortkey2(item, digit_place = 2):
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

    def get_sortkey2(i, digit_place=2):
        index = indexes[i]
        item = tlist[index]
        k = key(item)
        return (k// 10 ** digit_place) % 10

    for i in range(k):
        indexes = counting_sort_indexes(len(tlist) if tlist else 0, indexes, 10, partial(get_sortkey2, digit_place=i))

    return indexes

