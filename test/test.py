def plot(args):
    print(args.directory, args.do_cnmfe,args.motion_correction)

def main():
    import argparse
    parser = argparse.ArgumentParser() # fromfile_prefix_chars='@'

    parser.add_argument("--directory",type=str)
    parser.add_argument("--do_cnmfe",action='store_true')
    parser.add_argument("--motion_correction", action='store_true')

    args = parser.parse_known_args()[0]
    plot(args)

if __name__ == "__main__":
    main()