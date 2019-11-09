#!/usr/bin/env python
# coding=utf-8


class CPrint:
    '''
    Color Print:
        Print with beautiful color.
    Args:
        content:    The string will be showed.
        kind:       The kind of content. Type: string.
        fg_color:   The color of content. Type: int or string.
        bg_color:   The color of background. Type: int or string.
        level:      The level of informations. If it is not None,
                        `kind` and `fg_color` and `bg_color` will loss efficacy.
        **kwargs:   Other keyword args will be passed to print function.
    ===== Color Table =====
    +--------+--------+--------+--------+
    | color  | name   | fg_num | bg_num |
    +--------+--------+--------+--------+
    | black  | 黑色   | 30     | 40     |
    | red    | 红色   | 31     | 41     |
    | green  | 绿色   | 32     | 42     |
    | yellow | 黄色   | 33     | 43     |
    | blue   | 蓝色   | 34     | 44     |
    | purple | 紫红色 | 35     | 45     |
    | cyan   | 青蓝色 | 36     | 46     |
    | white  | 白色   | 37     | 47     |
    +--------+--------+--------+--------+
    ===== Kind Table =====
    +------+----------+
    | kind | effect   |
    +------+----------+
    | 0    | 终端默认 |
    | 1    | 高亮     |
    | 4    | 下划线   |
    | 5    | 闪烁     |
    | 7    | 反白     |
    | 8    | 不可见   |
    | 22   | 非粗体   |
    | 24   | 非下划线 |
    | 25   | 非闪烁   |
    | 27   | 非反显   |
    +------+----------+
    '''

    # control whether print colorful
    print_color = True

    _base_string = '\033[{}m{}\033[0m'
    _COLOR2NUM = {
        'black':  [30, 40],
        'red':    [31, 41],
        'green':  [32, 42],
        'yellow': [33, 43],
        'blue':   [34, 44],
        'purple': [35, 45],
        'cyan':   [36, 46],
        'white':  [37, 47]
    }
    _LEVEL2STYLE = {
        'debug': '0;32',
        'DEBUG': '0;32',
        'info':  '0;37',
        'INFO':  '0;37',
        'warn':  '0;37;43',
        'WARN':  '0;37;43',
        'error': '0;32;41',
        'ERROR': '0;32;41',
        'fatal': '5;32;41',
        'FATAL': '5;32;41'
    }

    def __call__(self, content, kind=None, fg_color=None, bg_color=None, level=None, **kwargs):
        style = ''

        if level is not None:
            style = self._LEVEL2STYLE[level]
        elif not any([kind, fg_color, bg_color]):
            style = self._LEVEL2STYLE['info']
        else:
            if kind is not None:
                assert isinstance(kind, int), 'The type of kind must be int.'
                style += str(kind) + ';'
            # fg
            if fg_color is not None:
                assert isinstance(fg_color, int) or fg_color in self._COLOR2NUM, \
                    f"fg_color must be int or one of {list(self._COLOR2NUM.keys())}, but got '{fg_color}'"
                if isinstance(fg_color, str):
                    fg_color = self._COLOR2NUM[fg_color][0]
                style += str(fg_color) + ';'
            # bg
            if bg_color is not None:
                assert isinstance(bg_color, int) or bg_color in self._COLOR2NUM, \
                    f"bg_color must be int or one of {list(self._COLOR2NUM.keys())}, but got '{bg_color}'"
                if isinstance(bg_color, str):
                    bg_color = self._COLOR2NUM[bg_color][1]
                style += str(bg_color)
            # remove `;` in end
            if style[-1] == ';':
                style = style[:-1]

        print(self._base_string.format(style, content), **kwargs)


cprint = CPrint()

if __name__ == '__main__':
    print(cprint.__doc__)
    print('test color:')
    cprint('Test test test Test', kind=0, fg_color='white', bg_color='yellow')
    cprint('Test test test Test', kind=1, fg_color='green', bg_color='red')
    cprint('Test test test Test', kind=4, fg_color='yellow', bg_color='blue')
    cprint('Test test test Test', kind=5, fg_color='purple', bg_color='cyan')
    
    print('\ntest level:')
    cprint('Test test test Test', level='debug')
    cprint('Test test test Test', level='info')
    cprint('Test test test Test', level='warn')
    cprint('Test test test Test', level='error')
    cprint('Test test test Test', level='fatal')
