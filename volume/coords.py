"""Logic for dealing with coordinates.

This introduces some helpers and terminology that are used throughout Minigo.

Minigo Coordinate: This is a tuple of the form (row, column) that is indexed
    starting out at (0, 0) from the upper-left.
Flattened Coordinate: this is a number ranging from 0 - N^2 (so N^2+1
    possible values). The extra value N^2 is used to mark a 'pass' move.
SGF Coordinate: Coordinate used for SGF serialization format. Coordinates use
    two-letter pairs having the form (column, row) indexed from the upper-left
    where 0, 0 = 'aa'.
GTP Coordinate: Human-readable coordinate string indexed from bottom left, with
    the first character a capital letter for the column and the second a number
    from 1-19 for the row. Note that GTP chooses to skip the letter 'I' due to
    its similarity with 'l' (lowercase 'L').
PYGTP Coordinate: Tuple coordinate indexed starting at 1,1 from bottom-left
    in the format (column, row)

So, for a 19x19,

Coord Type      upper_left      upper_right     pass
-------------------------------------------------------
minigo coord    (0, 0)          (0, 18)         None
flat            0               18              361
SGF             'aa'            'sa'            ''
GTP             'A19'           'T19'           'pass'
"""

import go

# We provide more than 19 entries here in case of boards larger than 19 x 19.
_SGF_COLUMNS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_GTP_COLUMNS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'


def from_flat(flat):
    """Converts from a flattened coordinate to a Minigo coordinate."""
    if flat == go.N * go.N:
        return None
    return divmod(flat, go.N)


def to_flat(coord):
    """Converts from a Minigo coordinate to a flattened coordinate."""
    if coord is None:
        return go.N * go.N
    return go.N * coord[0] + coord[1]


def from_sgf(sgfc):
    """Converts from an SGF coordinate to a Minigo coordinate."""
    if sgfc is None or sgfc == '' or (go.N <= 19 and sgfc == 'tt'):
        return None
    return _SGF_COLUMNS.index(sgfc[1]), _SGF_COLUMNS.index(sgfc[0])


def to_sgf(coord):
    """Converts from a Minigo coordinate to an SGF coordinate."""
    if coord is None:
        return ''
    return _SGF_COLUMNS[coord[1]] + _SGF_COLUMNS[coord[0]]


def from_gtp(gtpc):
    """Converts from a GTP coordinate to a Minigo coordinate."""
    gtpc = gtpc.upper()
    if gtpc == 'PASS':
        return None
    col = _GTP_COLUMNS.index(gtpc[0])
    row_from_bottom = int(gtpc[1:])
    return go.N - row_from_bottom, col


def to_gtp(coord):
    """Converts from a Minigo coordinate to a GTP coordinate."""
    if coord is None:
        return 'pass'
    y, x = coord
    return '{}{}'.format(_GTP_COLUMNS[x], go.N - y)