import argparse
from datetime import datetime


time_format = "%Y-%m-%d %H:%M:%S"


def get_time(prompt):
    time = None
    while time is None:
        line = raw_input(prompt)
        line = line.strip("\n ")

        # Try splitting up the time format
        try:
            parsed_time = datetime.strptime(line, time_format)
        except ValueError:
            print "Invalid time format"
        else:
            return parsed_time

    return time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple time calculator")
    opts = parser.parse_args()

    print "Time format: YYYY-MM-DD hh:mm:ss"
    start_time = get_time("Start time: ")
    end_time = get_time("End time: ")

    time_diff = end_time - start_time
    print time_diff