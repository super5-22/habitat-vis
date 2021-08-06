INTERACTION_MAPPING = \
    {
        ord('w'): "move_forward",
        ord('s'): "move_backward",
        ord('a'): "turn_left",
        ord('d'): "turn_right",
        83: "rotate_sensor_anti_clockwise",
        81: "rotate_sensor_clockwise",
        82: "look_up",
        84: "look_down",

        # direction button cannot work after clicking the image
        ord('l'): "rotate_sensor_anti_clockwise",
        ord('h'): "rotate_sensor_clockwise",
        ord('j'): "look_up",
        ord('k'): "look_down",

        # For WaitKeyEx
        65363: "rotate_sensor_anti_clockwise",
        65361: "rotate_sensor_clockwise",
        65362: "look_up",
        65364: "look_down",
    }

MODE_MAPPING = \
    {
        ord('u'): "user_mode",
        ord('p'): "policy_mode",
        ord('c'): "change_display",
    }
