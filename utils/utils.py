from webcolors import (
    rgb_to_name,
)


def convert_rgb_to_names(rgb_tuple: tuple[int, int, int]) -> str:
    return rgb_to_name(rgb_tuple)
