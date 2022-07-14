#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import time
from typing import Optional

text_colors = {
    'logs': '\033[34m',  # 033 is the escape code and 34 is the color code
    'info': '\033[32m',
    'warning': '\033[33m',
    'error': '\033[31m',
    'bold': '\033[1m',
    'end_color': '\033[0m',
    'light_red': '\033[36m'
}

class logger(object):
    def __init__(self):
        self.title = 'Logger'

    @classmethod
    def get_curr_time_stamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def error(self, message: str) -> None:
        time_stamp = self.get_curr_time_stamp()
        error_str = text_colors['error'] + text_colors['bold'] + 'ERROR  ' + text_colors['end_color']
        print('{} - {} - {}'.format(time_stamp, error_str, message))
        print('{} - {} - {}'.format(time_stamp, error_str, 'Exiting!!!'))
        exit(-1)

    @classmethod
    def color_text(self, in_text: str) -> str:
        return text_colors['light_red'] + in_text + text_colors['end_color']

    @classmethod
    def log(self, message: str) -> None:
        time_stamp = self.get_curr_time_stamp()
        log_str = text_colors['logs'] + text_colors['bold'] + 'LOGS   ' + text_colors['end_color']
        print('{} - {} - {}'.format(time_stamp, log_str, message))

    @classmethod
    def warning(self, message: str) -> None:
        time_stamp = self.get_curr_time_stamp()
        warn_str = text_colors['warning'] + text_colors['bold'] + 'WARNING' + text_colors['end_color']
        print('{} - {} - {}'.format(time_stamp, warn_str, message))

    @classmethod
    def info(self, message: str, print_line: Optional[bool] = False) -> None:
        time_stamp = self.get_curr_time_stamp()
        info_str = text_colors['info'] + text_colors['bold'] + 'INFO   ' + text_colors['end_color']
        print('{} - {} - {}'.format(time_stamp, info_str, message))
        if print_line:
            self.double_dash_line(dashes=150)

    @classmethod
    def double_dash_line(self, dashes: Optional[int] = 75) -> None:
        print(text_colors['error'] + '=' * dashes + text_colors['end_color'])

    @classmethod
    def singe_dash_line(self, dashes: Optional[int] = 67) -> None:
        print('-' * dashes)

    @classmethod
    def print_header(self, header: str) -> None:
        self.double_dash_line()
        print(text_colors['info'] + text_colors['bold'] + '=' * 50 + str(header) + text_colors['end_color'])
        self.double_dash_line()


    def print_header_minor(header: str) -> None:
        print(text_colors['warning'] + text_colors['bold'] + '=' * 25 + str(header) + text_colors['end_color'])

