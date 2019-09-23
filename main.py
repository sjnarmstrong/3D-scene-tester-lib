def get_config_file():
    from glob import glob
    config_files = glob("configs/*.yaml")
    config_files.extend(glob("user_configs/*.yaml"))
    i_start = 0
    while True:
        print("Please select one of the following config files:")
        lines = [f"{i}) {file}" for i, file in enumerate(config_files[i_start: i_start+10])]
        allow_next = False
        allow_prev = False
        if i_start > 0:
            lines.append("]) Next page")
            allow_next = True
        if i_start+10 <= len(config_files):
            lines.append("[) Previous page")
            allow_prev = True
        print('\n'.join(lines))
        ret_val = input("Please input your selection: ")
        if ret_val == ']':
            if allow_next:
                i_start += 10
            else:
                print("You are already on the first page")
        elif ret_val == '[':
            if allow_prev:
                i_start -= 10
            else:
                print("You are already on the last page")
        else:
            try:
                selection = int(ret_val)
                assert 0 <= selection <= 9
                return config_files[selection+i_start]
            except (ValueError, AssertionError, IndexError):
                print(f"Please enter a valid number between 0 and {len(config_files)-i_start}.")


def main(config_file):
    from segtester.configs.runnablebase import RunnableConfig
    assessment_config = RunnableConfig().parse_from_yaml_file(config_file)
    assessment_config()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run tests from a config file')
    parser.add_argument('--config_file', default=None)

    args = parser.parse_args()

    _config_file = get_config_file() if args.config_file is None else args.config_file
    main(_config_file)
