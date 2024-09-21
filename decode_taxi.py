"""
Function for decoding an encoding of state
in the environment Taxi.

"""


def decode(state: int):
    """
    Encoding of state -> taxi_row, taxi_col, passenger_location, destination


    ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

    Args:
        state (int): the state encoding (0 to 199)

    Returns:
        the parameters of the state (ints)
    """
    temp_state, destination = divmod(state, 4)
    temp_state, passenger_location = divmod(temp_state, 5)
    taxi_row, taxi_col = divmod(temp_state, 5)
    return taxi_row, taxi_col, passenger_location, destination


def to_str(state: int):
    row, col, passenger_location, destination = decode(state)
    return ("("
            + str(row) + "," + str(col)
            + ") - passenger: " + str(passenger_location)
            + " - destination: " + str(destination))
