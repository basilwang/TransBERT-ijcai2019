import os
from shutil import rmtree
import struct
from time import sleep


def nfs_rmtree(path, attempts=10):
    # Keep trying until the error goes away or we reached the limit
    for attempt in range(attempts):
        if not os.path.exists(path):
            return
        try:
            rmtree(path)
        except OSError as e:
            if e.errno == 16 and os.path.basename(e.filename).startswith(".nfs"):
                # This is an NFS file that couldn't be removed: wait and try again
                sleep(0.1)
            elif e.errno == 39:
                # Directory not empty: shouldn't happen with rmtree unless the file deletion didn't work: try again
                pass
            else:
                # Different error: proceed as usual
                raise


def ignore_comment_lines(input_iter, comment_char="#"):
    """
    Filter to remove lines commented out by a '#', or other comment character. Can
    be applied to a file to iterate over non-comment lines.

    """
    for line in input_iter:
        if not line.strip().startswith(comment_char):
            yield line


def split_sections(input_iter, section_heads):
    """
    Divide up the lines of a file by searching for the given section heads (whole lines) in the
    order they're given.

    A list of the lines of each section is returned in a list, in the same order as the given section
    head list.

    If the given heads are not all found, None values are returned in place of those sections
    (which will be at the end of the list). The number of returned sections will always be
    len(section_heads)+1 -- an extra one for the text before the first head.

    Note that, although this is designed for use with lines of text, there's nothing about it specific
    to text: the objects in the list (and section head list) could be of any type.

    """
    input_iter = iter(input_iter)
    section_heads = iter(section_heads)
    # basilwang 2020-10-21 python3 we use next()
    # next_head = section_heads.next()
    next_head = next(section_heads)
    sections = [[]]

    try:
        for line in input_iter:
            if line == next_head:
                # Move onto the next section
                sections.append([])
                # basilwang 2020-10-21 python3 we use next()
                # next_head = section_heads.next()
                next_head = next(section_heads)
            else:
                # Add this line to the current section
                sections[-1].append(line)
    except StopIteration:
        # Reached the end of the list of section names: include the remainder of the input in the last section
        sections[-1].extend(list(input_iter))

    # Pad out the sections if there are heads we didn't use
    remaining_heads = list(input_iter)
    if remaining_heads:
        sections.extend([None] * len(remaining_heads))

    return sections


def pickle_attrs(obj, attr_list, output_filename):
    """
    Pickle all the given attrs of the object in a single file, as a dictionary keyed by the
    attr names.

    :param obj: object to get attrs from
    :param attr_list: list of attr names
    :param output_filename: filename to pickle to
    """
    import cPickle as pickle
    data = dict([(attr_name, getattr(obj, attr_name)) for attr_name in attr_list])
    with open(output_filename, 'w') as output_file:
        pickle.dump(data, output_file)


class IntListsWriter(object):
    def __init__(self, filename):
        self.filename = filename
        self.writer = open(filename, 'w')

    def write(self, lst):
        self.writer.write("%s\n" % ",".join(str(num) for num in lst))

    def close(self):
        self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self


class IntListsReader(object):
    def __init__(self, filename):
        self.filename = filename

    def lists(self):
        with open(self.filename, 'r') as reader:
            for line in reader:
                # Remove the line break at the end
                line = line[:-1]
                # Catch the empty case
                if line:
                    yield [int(val) if val != "None" else None for val in line.split(",")]
                else:
                    yield []

    def __iter__(self):
        return self.lists()


class GroupedIntListsWriter(object):
    def __init__(self, filename):
        self.filename = filename
        self.writer = open(filename, 'w')

    def write(self, lsts):
        self.writer.write("%s\n" % " / ".join(",".join(str(num) for num in lst) for lst in lsts))

    def close(self):
        self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self


class GroupedIntListsReader(object):
    def __init__(self, filename):
        self.filename = filename
        self._length = None

    def __iter__(self):
        with open(self.filename, 'r') as reader:
            for line in reader:
                # Remove the line break at the end
                line = line.strip("\n ")
                # Catch the empty case
                if line:
                    yield [[int(val) if val != "None" else None for val in lst.split(",") if len(val)]
                           for lst in line.split(" / ")]
                else:
                    yield []

    def __len__(self):
        if self._length is None:
            with open(self.filename, 'r') as reader:
                self._length = sum(1 for __ in reader)
        return self._length


class GroupedIntTuplesReader(GroupedIntListsReader):
    def __iter__(self):
        for grp in GroupedIntListsReader.__iter__(self):
            yield [tuple(lst) for lst in grp]


BYTE_FORMATS = {
    # (num bytes, signed)
    (1, True): "b",   # signed char
    (1, False): "B",  # unsigned char
    (2, True): "h",   # signed short
    (2, False): "H",  # unsigned short
    (4, True): "l",   # signed long
    (4, False): "L",  # unsigned long
    (8, True): "q",   # signed long long
    (8, False): "Q",  # unsigned long long
}

def get_struct(bytes, signed, row_length):
    # Put together the formatting string for converting ints to bytes
    if (bytes, signed) not in BYTE_FORMATS:
        raise ValueError("invalid specification for int format: signed=%s, bytes=%s. signed must be bool, "
                         "bytes in [1, 2, 4, 8]" % (signed, bytes))
    format_string = "<" + BYTE_FORMATS[(bytes, signed)] * row_length
    # Compile the format for faster encoding
    return struct.Struct(format_string)


class IntTableWriter(object):
    """
    Similar to IntListsWriter, but every line has the same number of ints in it. This allows a more
    compact representation, which doesn't require converting the ints to strings or scanning for line ends,
    so is quite a bit quicker and results in much smaller file sizes.
    The downside is that the files are not human-readable.

    By default, the ints are stored as C longs, which use 4 bytes. If you know you don't need ints this
    big, you can choose 1 or 2 bytes, or even 8 (long long). By default, the ints are unsigned, but they
    may be signed.

    """
    def __init__(self, filename, row_length, signed=False, bytes=4):
        self.bytes = bytes
        self.signed = signed
        self.row_length = row_length
        self.filename = filename
        self.writer = open(filename, 'w')

        # Prepare a struct for efficiently encoding int rows as bytes
        self.struct = get_struct(bytes, signed, row_length)
        # Write the first few bytes of the file to denote the representation format
        self.writer.write(struct.pack("B?H", bytes, signed, row_length))

    def write(self, row):
        try:
            self.writer.write(self.struct.pack(*row))
        except struct.error as e:
            # Instead of checking the rows before encoding, catch any encoding errors and give helpful messages
            if len(row) != self.row_length:
                raise ValueError("tried to write a row of length %d to a table writer with row length %d" %
                                 (len(row), self.row_length))
            else:
                raise ValueError("error encoding int row %s using struct format %s: %s" %
                                 (row, self.struct.format, e))

    def close(self):
        self.writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self


class IntTableReader(object):
    def __init__(self, filename):
        self.filename = filename

    def lists(self):
        with open(self.filename, 'r') as reader:
            # Read the first few bytes of the file to establish the format for reading the rest
            bytes, signed, row_length = struct.unpack("B?H", reader.read(struct.calcsize("B?H")))
            # Compile a struct for unpacking these quickly
            unpacker = get_struct(bytes, signed, row_length)
            row_size = unpacker.size

            while True:
                # Read data for a single row
                row_string = reader.read(row_size)
                if row_string == "":
                    # Reach end of file
                    break
                try:
                    row = unpacker.unpack(row_string)
                except struct.error as e:
                    if len(row_string) < row_size:
                        # Got a partial row at end of file
                        raise IOError("found partial row at end of file: last row has byte length %d, not %d" %
                                      (len(row_string), row_size))
                    else:
                        raise IOError("error interpreting row: %s" % e)
                yield row

    def __iter__(self):
        return self.lists()
