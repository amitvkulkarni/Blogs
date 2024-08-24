import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"{'-'*20} Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds {'-'*20}"
        )
        return result

    return wrapper


def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")

    return wrapper


def set_frame_style(df, caption="", font_size="20px"):
    random_list = [
        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ]
    """Helper function to set dataframe presentation style.
    """
    return (
        df.style.background_gradient(cmap=random_list[np.random.randint(1, 17)])
        .set_caption(caption)
        .set_table_styles(
            [
                {
                    "selector": "caption",
                    "props": [
                        ("color", "Brown"),
                        ("font-size", font_size),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )


def highlight_max(s):
    is_max = s == s.max()
    return ["background-color: yellow; color: black" if v else "" for v in is_max]
